import gunpowder as gp
import numpy as np
import time
import logging
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import convolve
from numpy.lib.stride_tricks import as_strided

logger = logging.getLogger(__name__)

def get_local_shape_descriptors(
        segmentation,
        sigma,
        voxel_size=None,
        roi=None,
        labels=None,
        mode='gaussian',
        downsample=1):
    '''
    Compute local shape descriptors for the given segmentation.

    Args:

        segmentation (``np.array`` of ``int``):

            A label array to compute the local shape descriptors for.

        sigma (``tuple`` of ``float``):

            The radius to consider for the local shape descriptor.

        voxel_size (``tuple`` of ``int``, optional):

            The voxel size of ``segmentation``. Defaults to 1.

        roi (``gunpowder.Roi``, optional):

            Restrict the computation to the given ROI.

        labels (array-like of ``int``, optional):

            Restrict the computation to the given labels. Defaults to all
            labels inside the ``roi`` of ``segmentation``.

        mode (``string``, optional):

            Either ``gaussian`` or ``sphere``. Determines over what region
            the local shape descriptor is computed. For ``gaussian``, a
            Gaussian with the given ``sigma`` is used, and statistics are
            averaged with corresponding weights. For ``sphere``, a sphere
            with radius ``sigma`` is used. Defaults to 'gaussian'.

        downsample (``int``, optional):

            Compute the local shape descriptor on a downsampled volume for
            faster processing. Defaults to 1 (no downsampling).
    '''
    return LsdExtractor(sigma, mode, downsample).get_descriptors(
        segmentation,
        voxel_size,
        roi,
        labels)

class LsdExtractor(object):

    def __init__(self, sigma, mode='gaussian', downsample=1):
        '''
        Create an extractor for local shape descriptors. The extractor caches
        the data repeatedly needed for segmentations of the same size. If this
        is not desired, `func:get_local_shape_descriptors` should be used
        instead.

        Args:

            sigma (``tuple`` of ``float``):

                The radius to consider for the local shape descriptor.

            mode (``string``, optional):

                Either ``gaussian`` or ``sphere``. Determines over what region
                the local shape descriptor is computed. For ``gaussian``, a
                Gaussian with the given ``sigma`` is used, and statistics are
                averaged with corresponding weights. For ``sphere``, a sphere
                with radius ``sigma`` is used. Defaults to 'gaussian'.

            downsample (``int``, optional):

                Compute the local shape descriptor on a downsampled volume for
                faster processing. Defaults to 1 (no downsampling).
        '''
        self.sigma = sigma
        self.mode = mode
        self.downsample = downsample
        self.coords = {}

    def get_descriptors(
            self,
            segmentation,
            voxel_size=None,
            roi=None,
            labels=None):
        '''Compute local shape descriptors for a given segmentation.

        Args:

            segmentation (``np.array`` of ``int``):

                A label array to compute the local shape descriptors for.

            voxel_size (``tuple`` of ``int``, optional):

                The voxel size of ``segmentation``. Defaults to 1.

            roi (``gunpowder.Roi``, optional):

                Restrict the computation to the given ROI in voxels.

            labels (array-like of ``int``, optional):

                Restrict the computation to the given labels. Defaults to all
                labels inside the ``roi`` of ``segmentation``.
        '''

        dims = len(segmentation.shape)

        if voxel_size is None:
            voxel_size = gp.Coordinate((1,)*dims)
        else:
            voxel_size = gp.Coordinate(voxel_size)

        if roi is None:
            roi = gp.Roi((0,)*dims, segmentation.shape)

        roi_slices = roi.to_slices()

        if labels is None:
            labels = np.unique(segmentation[roi_slices])

        # prepare full-res descriptor volumes for roi
        descriptors = np.zeros((10,) + roi.get_shape(), dtype=np.float32)

        # get sub-sampled shape, roi, voxel size and sigma
        df = self.downsample
        logger.debug(
            "Downsampling segmentation %s with factor %f",
            segmentation.shape, df)
        sub_shape = tuple(s/df for s in segmentation.shape)
        sub_roi = roi/df
        assert sub_roi*df == roi, (
            "Segmentation shape %s is not a multiple of downsampling factor "
            "%d (sub_roi=%s, roi=%s)."%(
                segmentation.shape, self.downsample,
                sub_roi, roi))
        sub_voxel_size = tuple(v*df for v in voxel_size)
        sub_sigma_voxel = tuple(s/v for s, v in zip(self.sigma, sub_voxel_size))
        logger.debug("Downsampled shape: %s", sub_shape)
        logger.debug("Downsampled voxel size: %s", sub_voxel_size)
        logger.debug("Sigma in voxels: %s", sub_sigma_voxel)

        # prepare coords volume (reuse if we already have one)
        if (sub_shape, sub_voxel_size) not in self.coords:

            logger.debug("Create meshgrid...")

            self.coords[(sub_shape, sub_voxel_size)] = np.array(
                np.meshgrid(
                    np.arange(0, sub_shape[0]*sub_voxel_size[0], sub_voxel_size[0]),
                    np.arange(0, sub_shape[1]*sub_voxel_size[1], sub_voxel_size[1]),
                    np.arange(0, sub_shape[2]*sub_voxel_size[2], sub_voxel_size[2]),
                    indexing='ij'),
                dtype=np.float32)

        coords = self.coords[(sub_shape, sub_voxel_size)]

        # for all labels
        for label in labels:

            if label == 0:
                continue

            logger.debug("Creating shape descriptors for label %d", label)

            mask = (segmentation==label).astype(np.float32)
            logger.debug("Label mask %s", mask.shape)
            sub_mask = mask[::df, ::df, ::df]
            logger.debug("Downsampled label mask %s", sub_mask.shape)

            sub_count, sub_mean_offset, sub_variance, sub_pearson = self.__get_stats(
                coords,
                sub_mask,
                sub_sigma_voxel,
                sub_roi)

            sub_descriptor = np.concatenate([
                sub_mean_offset,
                sub_variance,
                sub_pearson,
                sub_count[None,:]])

            logger.debug("Upscaling descriptors...")
            start = time.time()
            descriptor = self.__upsample(sub_descriptor, df)
            logger.debug("%f seconds", time.time() - start)

            logger.debug("Accumulating descriptors...")
            start = time.time()
            descriptors += descriptor*mask[roi_slices]
            logger.debug("%f seconds", time.time() - start)

        # normalize stats

        # get max possible mean offset for normalization
        if self.mode == 'gaussian':
            # farthes voxel in context is 3*sigma away, but due to Gaussian
            # weighting, sigma itself is probably a better upper bound
            max_distance = np.array(
                [s for s in self.sigma],
                dtype=np.float32)
        elif self.mode == 'sphere':
            # farthest voxel in context is sigma away, but this is almost
            # impossible to reach as offset -- let's take half sigma
            max_distance = np.array(
                [0.5*s for s in self.sigma],
                dtype=np.float32)

        # mean offsets in [0, 1]
        descriptors[[0, 1, 2]] = descriptors[[0, 1, 2]]/max_distance[:, None, None, None]*0.5 + 0.5
        # pearsons in [0, 1]
        descriptors[[6, 7, 8]] = descriptors[[6, 7, 8]]*0.5 + 0.5
        # reset background to 0
        descriptors[[0, 1, 2, 6, 7, 8]] *= (segmentation[roi_slices] != 0)

        # clip outliers
        np.clip(descriptors, 0.0, 1.0, out=descriptors)

        # convert to uint8
        descriptors = (descriptors * 255).astype(np.uint8)

        return descriptors

    def __get_stats(self, coords, mask, sigma_voxel, roi):

        # mask for object
        masked_coords = coords*mask

        # number of inside voxels
        logger.debug("Counting inside voxels...")
        start = time.time()
        count = self.__aggregate(mask, sigma_voxel, self.mode, roi)
        # avoid division by zero
        count[count==0] = 1
        logger.debug("%f seconds", time.time() - start)

        # mean
        logger.debug("Computing mean position of inside voxels...")
        start = time.time()
        mean = np.array([
            self.__aggregate(masked_coords[d], sigma_voxel, self.mode, roi)
            for d in range(3)])
        mean /= count
        logger.debug("%f seconds", time.time() - start)

        logger.debug("Computing offset of mean position...")
        start = time.time()
        mean_offset = mean - coords[(slice(None),) + roi.to_slices()]

        # covariance
        logger.debug("Computing covariance...")
        coords_outer = self.__outer_product(masked_coords)
        covariance = np.array([
            self.__aggregate(coords_outer[d], sigma_voxel, self.mode, roi)
            # remove duplicate entries in covariance
            # 0 1 2
            # 3 4 5
            # 6 7 8
            for d in [0, 4, 8, 1, 2, 5]])
        covariance /= count
        covariance -= self.__outer_product(mean)[[0, 4, 8, 1, 2, 5]]
        logger.debug("%f seconds", time.time() - start)

        # variances of z, y, x coordinates
        variance = covariance[[0, 1, 2]]
        # Pearson coefficients of zy, zx, yx
        pearson = covariance[[3, 4, 5]]

        # normalize Pearson correlation coefficient
        variance[variance<1e-3] = 1e-3 # numerical stability
        pearson[0] /= np.sqrt(variance[0]*variance[1])
        pearson[1] /= np.sqrt(variance[0]*variance[2])
        pearson[2] /= np.sqrt(variance[1]*variance[2])

        # normalize variances to interval [0, 1]
        variance[0] /= self.sigma[0]**2
        variance[1] /= self.sigma[1]**2
        variance[2] /= self.sigma[2]**2

        return count, mean_offset, variance, pearson

    def __make_sphere(self, radius):

        logger.debug("Creating sphere with radius %d...", radius)

        r2 = np.arange(-radius, radius)**2
        dist2 = r2[:, None, None] + r2[:, None] + r2
        return (dist2 <= radius**2).astype(np.float32)

    def __aggregate(self, array, sigma, mode='gaussian', roi=None):

        if roi is None:
            roi_slices = (slice(None),)
        else:
            roi_slices = roi.to_slices()

        if mode == 'gaussian':

            return gaussian_filter(
                array,
                sigma=sigma,
                mode='constant',
                cval=0.0,
                truncate=3.0)[roi_slices]

        elif mode == 'sphere':

            radius = sigma[0]
            for d in range(len(sigma)):
                assert radius == sigma[d], (
                    "For mode 'sphere', only isotropic sigma is allowed.")

            sphere = self.__make_sphere(radius)
            return convolve(
                array,
                sphere,
                mode='constant',
                cval=0.0)[roi_slices]

        else:
            raise RuntimeError("Unknown mode %s"%mode)

    def get_context(self):
        '''Return the context needed to compute the LSDs.'''
        if self.mode == 'gaussian':
            return tuple((3.0*s for s in self.sigma))
        elif self.mode == 'sphere':
            return self.sigma

    def __outer_product(self, array):
        '''Computes the unique values of the outer products of the first dimension
        of ``array``. If ``array`` has shape ``(k, d, h, w)``, for example, the
        output will be of shape ``(k*(k+1)/2, d, h, w)``.
        '''
        k = array.shape[0]
        outer = np.einsum('i...,j...->ij...', array, array)
        return outer.reshape((k**2,)+array.shape[1:])

    def __upsample(self, array, f):

        shape = array.shape
        stride = array.strides

        view = as_strided(
            array,
            (shape[0], shape[1], f, shape[2], f, shape[3], f),
            (stride[0], stride[1], 0, stride[2], 0, stride[3], 0))

        return view.reshape(shape[0], shape[1]*f, shape[2]*f, shape[3]*f)
