import math
from feature_utils import *
from samplers.uniform_sampler import *


class RANSAC(object):

    def __init__(
            self,
            estimator,
            sampler,
            scoring,
            fmat=False,
            train=False,
            ransac_batch_size=64,
            sampler_id=0,
            weighted=0,
            threshold=1e-3,
            confidence=0.999,
            max_iterations=5000,
            lo=0,#2,
            lo_iters=64,
            eps=1e-5
    ):

        self.estimator = estimator
        self.sampler = sampler
        self.scoring = scoring
        self.lo = lo
        self.lo_iters = lo_iters
        self.fmat = fmat
        self.train = train
        self.ransac_batch_size = ransac_batch_size
        self.sampler_id = sampler_id
        self.weighted = weighted
        self.threshold = threshold
        self.confidence = confidence
        self.max_iterations = max_iterations
        self.eps = eps

    def __call__(self, matches, logits, K1, K2, gt_model):

        iterations = 0
        best_score = 0
        point_number = matches.shape[0]
        best_mask = []
        best_model = []
        models = {}
        if self.fmat:
            normalized_multipler = 1
        else:
            normalized_multipler = (K1[0, 0] + K1[1, 1] + K1[0, 0] + K2[1, 1]) / 4
        threshold = self.threshold / normalized_multipler
        max_iters = self.max_iterations
        while iterations < max_iters:

            # Select minimal samples for the current batch, GumbelSoftmax Sampler (id=2) can propagate the gradients
            if self.sampler_id != 2 and self.sampler_id != 3:
                minimal_sample_indices = self.sampler.sample()
                minimal_samples = matches[minimal_sample_indices]
                samples, soft_weights = None, None
            else:
                samples, soft_weights = self.sampler.sample(logits)
                points = matches.repeat([self.ransac_batch_size, 1, 1]) * samples.unsqueeze(-1)
                minimal_samples = points[samples != 0].view(self.ransac_batch_size, -1, matches.shape[-1])
            # when there is no minimal sample comes, skip
            if minimal_samples.shape[1] == 0:
                continue
            # Estimate models' parameters, can propagate gradient
            if self.weighted:
                estimated_models = self.estimator.estimate_model(
                    minimal_samples,
                    soft_weights[samples != 0].view(self.ransac_batch_size, -1)
                )
            else:
                estimated_models = self.estimator.estimate_model(minimal_samples)

            if self.train:
                # for learning, return all models and sum the pose errors of all models instead of selecting the best
                # choose the best model from each sample, in the case of generating more than one models from the sample
                if estimated_models.shape[0] == 0 or estimated_models is None:
                    continue

                if self.sampler.num_samples == 8:
                    chosen_models = estimated_models
                else:
                    solution_num = 4 if self.fmat else 10
                    distances = torch.norm(estimated_models - gt_model, dim=(1, 2)).view(estimated_models.shape[0], -1)
                    try:
                        chosen_indices = torch.argmin(distances.view(-1, solution_num), dim=-1)
                        chosen_models = torch.stack(
                          [
                              (estimated_models.view(-1, solution_num, 3, 3))[i, chosen_indices[i], :]
                              for i in range(int(estimated_models.shape[0] / solution_num))
                          ]
                        )

                    except ValueError as e:
                        print("not enough models for selection, we choose the first solution in this batch",
                              e, estimated_models.shape)
                        chosen_models = estimated_models[0].unsqueeze(0)

                if torch.isnan(chosen_models).any():
                    # deal with the error of linalg.slove,
                    # "The diagonal element 1 is zero, the solver could not completed because the input singular matrix)
                    print("Delete those models having problems with singular matrix.")
                nan_filter = [not (torch.isnan(model).any()) for model in chosen_models]
                models[iterations] = chosen_models[torch.as_tensor(nan_filter)]
            else:
                # Calculate the scores of the models
                scores, inlier_masks = self.scoring.score(matches, estimated_models, threshold)

                # Select the best model
                best_idx = torch.argmax(scores)
                # Update the best model if this iteration is better
                if scores[best_idx] > best_score or iterations == 0:
                    best_score = scores[best_idx]
                    best_mask = inlier_masks[best_idx]
                    best_model = estimated_models[best_idx]
                    best_inlier_number = torch.sum(best_mask)
                    # Apply local optimization if needed
                    if self.lo:
                        best_score, best_mask, best_model, best_inlier_number = self.localOptimization(
                            best_score,
                            best_mask,
                            best_model,
                            best_inlier_number,
                            matches,
                            K1,
                            K2,
                            threshold
                        )

                    # use adaptive iteration number when testing, update the max iteration number by inlier counts
                    max_iters = min(
                        self.max_iterations,
                        self.adaptive_iteration_number(
                            best_inlier_number,
                            point_number,
                            self.confidence
                        )
                    )

            iterations += self.ransac_batch_size

        # not needed for learning, so no differentiability is needed
        # Final refitting on the inliers
        if not self.train:
            inlier_indices = best_mask.nonzero(as_tuple=True)
            inlier_points = matches[inlier_indices].unsqueeze(0)
            if self.fmat:
                if self.weighted:
                    estimated_models = self.estimator.estimate_model(inlier_points, soft_weights[0, inlier_indices[0]])
                else:
                    estimated_models = self.estimator.estimate_model(inlier_points)
            else:
                estimated_models = self.estimator.estimate_model(
                    matches.unsqueeze(0).double(),
                    K1=K1.cpu().detach().numpy(),
                    K2=K2.cpu().detach().numpy(),
                    inlier_indices=inlier_indices[0].cpu().detach().numpy().astype(np.uint64),
                    best_model=best_model.cpu().detach().numpy().T,
                    unnormalzied_threshold=0.75,
                    best_score=best_score
                )

            # Select the best if more than one models are returned
            if estimated_models is None:
                best_model = torch.eye(3, 3, device=best_model.device, dtype=best_model.dtype)
            elif estimated_models.shape[0] == 0:
                best_model = torch.eye(3, 3, device=estimated_models.device, dtype=estimated_models.dtype)

            elif estimated_models.shape[0] >= 1:
                if estimated_models.dtype != matches.dtype:
                    estimated_models = estimated_models.to(matches.dtype)
                #if estimated_models.type() == 'torch.cuda.DoubleTensor' or 'torch.DoubleTensor':
                    #estimated_models = estimated_models.to(torch.float)

                # Calculate the scores of the models
                scores, inlier_masks = self.scoring.score(matches, estimated_models, threshold)

                if max(scores) > best_score:
                    best_idx = torch.argmax(scores)
                    best_model = estimated_models[best_idx]
                    best_score = scores[best_idx]
            else:
                best_model = estimated_models[0]

            if not self.scoring.provides_inliers:
                best_model, best_mask = self.scoring.get_inliers(
                    matches,
                    best_model.unsqueeze(0),
                    self.estimator,
                    threshold=threshold
                )
        else:
            best_model = models
        # if best_model.shape[0] == 0:
        #     best_model = torch.eye(3, 3, device=best_model.device, dtype=best_model.dtype)
        return best_model, best_mask, best_score, iterations

    def adaptive_iteration_number(self, inlier_number, point_number, confidence):
        inlier_ratio = inlier_number / point_number
        probability = 1.0 - inlier_ratio ** self.estimator.sample_size
        if probability >= 1.0 - self.eps:
            return self.max_iterations

        try:
            max(0.0, (math.log10(1.0 - confidence) / (
                        math.log10(1 - inlier_ratio ** self.estimator.sample_size) + self.eps)))
        except ValueError:
            print("add eps to avoid math domain error of log", 1 - inlier_ratio ** self.estimator.sample_size, '\n')

        return max(0.0, (math.log10(1.0 - confidence) / (
            math.log10(1 - inlier_ratio ** self.estimator.sample_size + self.eps))))

    def localOptimization(self, best_score, best_mask, best_model, best_inlier_number, matches, K1, K2, threshold):

        # Do a single or iterated LSQ fitting
        if self.lo < 3:
            iters = 1
            if self.lo == 2:
                iters = self.lo_iters

            for iter_i in range(iters):
                # Select the inliers
                indices = best_mask.nonzero(as_tuple=True)
                points = torch.unsqueeze(matches[indices], 0)

                # Estimate the model from all points
                if self.fmat:
                    models = self.estimator.estimate_model(points)
                else:
                    models = self.estimator.estimate_model(
                        points,
                        K1=K1.cpu().detach().numpy(),
                        K2=K2.cpu().detach().numpy(),
                        inlier_indices=indices[0].cpu().detach().numpy().astype(np.uint64),
                        best_model=best_model.cpu().detach().numpy().T,
                        unnormalzied_threshold=0.75,
                        best_score=best_score
                    )
                if models is None:
                    models = torch.eye(3).unsqueeze(0).to(points.device)
                # Calculate the score
                scores, inlier_masks = self.scoring.score(matches, models, threshold)

                # Select the best model
                best_idx = torch.argmax(scores)

                if scores[best_idx] >= best_score:
                    best_score = scores[best_idx]
                    best_mask = inlier_masks[best_idx]
                    best_model = models[best_idx]
                    best_inlier_number = torch.sum(best_mask)
                else:
                    break
        elif self.lo == 3:  # Do inner RANSAC
            # Calculate the sample size
            sample_size = 7 * self.estimator.sample_size
            if best_inlier_number < sample_size:
                sample_size = self.estimator.sample_size

            # Initialize the LO sampler
            lo_sampler = UniformSampler(self.lo_iters, sample_size, matches.shape[0])

            for iter_i in range(self.lo_iters):
                # Select minimal samples for the current batch
                minimal_sample_indices = lo_sampler.sample()
                minimal_samples = matches[minimal_sample_indices]

                # Estimate the models' parameters
                estimated_models = self.estimator.estimate_model(minimal_samples)

                # Calculate the scores of the models
                scores, inlier_masks = self.scoring.score(matches, estimated_models, threshold)

                # Select the best model
                best_idx = torch.argmax(scores)

                # The loss should be: sum_{1}^k pose_error(model_k, model_gt) (where k is iteration number/batch size)
                # Update the previous best model if needed
                if scores[best_idx] > best_score:
                    best_score = scores[best_idx]
                    best_mask = inlier_masks[best_idx]
                    best_model = estimated_models[best_idx]
                    best_inlier_number = torch.sum(best_mask)

                    # Re-calculate the sample size
                    sample_size = 7 * self.estimator.sample_size
                    if best_inlier_number < sample_size:
                        sample_size = self.estimator.sample_size

                    # Re-initialize the LO sampler
                    lo_sampler = UniformSampler(self.ransac_batch_size, sample_size, matches.shape[0])
                else:
                    break

        return best_score, best_mask, best_model, best_inlier_number



class RANSAC3D(object):

    def __init__(
            self,
            estimator,
            sampler,
            scoring,
            fmat=False,
            train=False,
            ransac_batch_size=64,
            sampler_id=0,
            weighted=0,
            threshold=1e-3,
            confidence=0.999,
            max_iterations=5000,
            lo=0,#2,
            lo_iters=64,
            eps=1e-5
    ):

        self.estimator = estimator
        self.sampler = sampler
        self.scoring = scoring
        self.lo = lo
        self.lo_iters = lo_iters
        self.fmat = fmat
        self.train = train
        self.ransac_batch_size = ransac_batch_size
        self.sampler_id = sampler_id
        self.weighted = weighted
        self.threshold = threshold
        self.confidence = confidence
        self.max_iterations = max_iterations
        self.eps = eps

    def __call__(self, matches, logits, gt_model, valid=False):

        if valid:
            # import pdb; pdb.set_trace()
            self.train = False
        iterations = 0
        best_score = 0
        point_number = matches.shape[0]
        best_mask = []
        best_model = []
        mean_residuals = {}
        residuals = {}
        models = {}
        selected_indices = {}
        while iterations < self.max_iterations:

            # Select minimal samples for the current batch, GumbelSoftmax Sampler (id=2) can propagate the gradients
            if self.sampler_id != 2 and self.sampler_id != 3:
                minimal_sample_indices = self.sampler.sample()
                minimal_samples = matches[minimal_sample_indices]
                samples, soft_weights = None, None
            else:
                samples, soft_weights = self.sampler.sample(logits)
                points = matches.repeat([self.ransac_batch_size, 1, 1]) * samples.unsqueeze(-1)
                minimal_samples = points[samples != 0].view(self.ransac_batch_size, -1, matches.shape[-1])
            # when there is no minimal sample comes, skip
            if minimal_samples.shape[1] == 0:
                continue
            # Estimate models' parameters, can propagate gradient
            estimated_models, R, t, _ = self.estimator.estimate_model(minimal_samples)

            if self.train:
                if estimated_models.shape[0] == 0 or estimated_models is None:
                    continue
                if torch.isnan(estimated_models).any():
                    # deal with the error of linalg.slove,
                    # "The diagonal element 1 is zero, the solver could not completed because the input singular matrix)
                    print("Delete those models having problems with singular matrix.")
                nan_filter = [not (torch.isnan(model).any()) for model in estimated_models]
                models[iterations] = estimated_models[torch.as_tensor(nan_filter)]
                # selected_indices[iterations] = minimal_sample_indices#[torch.as_tensor(nan_filter)]

                residual, mean_residual, inlier_mask = self.estimator.squared_residual(matches[:, :3], matches[:, 3:], estimated_models[:, :3, :].transpose(-1, -2))
                residuals[iterations] = residual
                mean_residuals[iterations] = mean_residual
            else:
                selected_indices[iterations] = minimal_sample_indices

                # Calculate the residuals of the models
                # scores, inlier_masks = self.scoring.score(matches, estimated_models)
                residual, mean_residual, inlier_mask = self.estimator.squared_residual(matches[:, :3], matches[:, 3:], estimated_models[:, :3, :].transpose(-1, -2))
                # Select the best model
                best_idx = torch.argmax(scores)
                # Update the best model if this iteration is better
                if scores[best_idx] > best_score or iterations == 0:
                    best_score = scores[best_idx]
                    best_mask = inlier_masks[best_idx]
                    best_model = estimated_models[best_idx]
                    best_inlier_number = torch.sum(best_mask)                    

                    # use adaptive iteration number when testing, update the max iteration number by inlier counts
                    self.max_iterations = min(
                        self.max_iterations,
                        self.adaptive_iteration_number(
                            best_inlier_number,
                            point_number,
                            self.confidence
                        )
                    )

            iterations += self.ransac_batch_size

        # not needed for learning, so no differentiability is needed
        # Final refitting on the inliers
        if not self.train:
            inlier_indices = best_mask.nonzero(as_tuple=True)
            inlier_points = matches[inlier_indices].unsqueeze(0)
            estimated_models = self.estimator.estimate_model(inlier_points, soft_weights[0, inlier_indices[0]])

            # Select the best if more than one models are returned
            if estimated_models is None:
                best_model = torch.eye(3, 3, device=best_model.device, dtype=best_model.dtype)
            elif estimated_models.shape[0] == 0:
                best_model = torch.eye(3, 3, device=estimated_models.device, dtype=estimated_models.dtype)

            elif estimated_models.shape[0] >= 1:
                if estimated_models.dtype != matches.dtype:
                    estimated_models = estimated_models.to(matches.dtype)
                #if estimated_models.type() == 'torch.cuda.DoubleTensor' or 'torch.DoubleTensor':
                    #estimated_models = estimated_models.to(torch.float)

                # Calculate the scores of the models
                scores, inlier_masks = self.scoring.score(matches, estimated_models)

                if max(scores) > best_score:
                    best_idx = torch.argmax(scores)
                    best_model = estimated_models[best_idx]
                    best_score = scores[best_idx]
            else:
                best_model = estimated_models[0]

            if not self.scoring.provides_inliers:
                best_model, best_mask = self.scoring.get_inliers(
                    matches,
                    best_model.unsqueeze(0),
                    self.estimator,
                    threshold=self.threshold
                )
            
        else:
            best_model = models

        return best_model, residuals, mean_residuals, best_score, iterations

    def adaptive_iteration_number(self, inlier_number, point_number, confidence):
        inlier_ratio = inlier_number / point_number
        probability = 1.0 - inlier_ratio ** self.estimator.sample_size
        if probability >= 1.0 - self.eps:
            return self.max_iterations

        try:
            max(0.0, (math.log10(1.0 - confidence) / (
                        math.log10(1 - inlier_ratio ** self.estimator.sample_size) + self.eps)))
        except ValueError:
            print("add eps to avoid math domain error of log", 1 - inlier_ratio ** self.estimator.sample_size, '\n')

        return max(0.0, (math.log10(1.0 - confidence) / (
            math.log10(1 - inlier_ratio ** self.estimator.sample_size + self.eps))))

    def localOptimization(self, best_score, best_mask, best_model, best_inlier_number, matches, K1, K2):

        # Do a single or iterated LSQ fitting
        if self.lo < 3:
            iters = 1
            if self.lo == 2:
                iters = self.lo_iters

            for iter_i in range(iters):
                # Select the inliers
                indices = best_mask.nonzero(as_tuple=True)
                points = torch.unsqueeze(matches[indices], 0)

                # Estimate the model from all points
                if self.fmat:
                    models = self.estimator.estimate_model(points)
                else:
                    models = self.estimator.estimate_model(
                        points,
                        K1=K1.cpu().detach().numpy(),
                        K2=K2.cpu().detach().numpy(),
                        inlier_indices=indices[0].cpu().detach().numpy().astype(np.uint64),
                        best_model=best_model.cpu().detach().numpy().T,
                        unnormalzied_threshold=0.75,
                        best_score=best_score
                    )
                if models is None:
                    models = torch.eye(3).unsqueeze(0).to(points.device)
                # Calculate the score
                scores, inlier_masks = self.scoring.score(matches, models)

                # Select the best model
                best_idx = torch.argmax(scores)

                if scores[best_idx] >= best_score:
                    best_score = scores[best_idx]
                    best_mask = inlier_masks[best_idx]
                    best_model = models[best_idx]
                    best_inlier_number = torch.sum(best_mask)
                else:
                    break
        elif self.lo == 3:  # Do inner RANSAC
            # Calculate the sample size
            sample_size = 7 * self.estimator.sample_size
            if best_inlier_number < sample_size:
                sample_size = self.estimator.sample_size

            # Initialize the LO sampler
            lo_sampler = UniformSampler(self.lo_iters, sample_size, matches.shape[0])

            for iter_i in range(self.lo_iters):
                # Select minimal samples for the current batch
                minimal_sample_indices = lo_sampler.sample()
                minimal_samples = matches[minimal_sample_indices]

                # Estimate the models' parameters
                estimated_models = self.estimator.estimate_model(minimal_samples)

                # Calculate the scores of the models
                scores, inlier_masks = self.scoring.score(matches, estimated_models)

                # Select the best model
                best_idx = torch.argmax(scores)

                # The loss should be: sum_{1}^k pose_error(model_k, model_gt) (where k is iteration number/batch size)
                # Update the previous best model if needed
                if scores[best_idx] > best_score:
                    best_score = scores[best_idx]
                    best_mask = inlier_masks[best_idx]
                    best_model = estimated_models[best_idx]
                    best_inlier_number = torch.sum(best_mask)

                    # Re-calculate the sample size
                    sample_size = 7 * self.estimator.sample_size
                    if best_inlier_number < sample_size:
                        sample_size = self.estimator.sample_size

                    # Re-initialize the LO sampler
                    lo_sampler = UniformSampler(self.ransac_batch_size, sample_size, matches.shape[0])
                else:
                    break

        return best_score, best_mask, best_model, best_inlier_number