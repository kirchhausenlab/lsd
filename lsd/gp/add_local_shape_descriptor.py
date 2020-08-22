from __future__ import absolute_import
from ..local_shape_descriptor import LsdExtractor
from gunpowder import BatchFilter, Array
import logging
import numpy as np
import time

logger = logging.getLogger(__name__)

class AddLocalShapeDescriptor(BatchFilter):
    '''Create a local segmentation shape discriptor to each voxel.

    Args:

        segmentation (:class:`ArrayKey`): The array storing the segmentation
            to use.

        descriptor (:class:`ArrayKey`): The array of the shape descriptor to
            generate.

        mask (:class:`ArrayKey`, optional): The array to store a binary mask
            the size of the descriptors. Background voxels, which do not have a
            descriptor, will be set to 0. This can be used as a loss scale
            during training, such that background is ignored.

        sigma (float or tuple of float): The context to consider to compute
            the shape descriptor in world units. This will be the standard
            deviation of a Gaussian kernel or the radius of the sphere.

        mode (string): Either ``gaussian`` or ``sphere``. Specifies how to
            accumulate local statistics: ``gaussian`` uses Gaussian convolution
            to compute a weighed average of statistics inside an object.
            ``sphere`` accumulates values in a sphere.

        downsample (int, optional): Downsample the segmentation mask to extract
            the statistics with the given factore. Default is 1 (no
            downsampling).
    '''

    def __init__(
            self,
            segmentation,
            descriptor,
            mask=None,
            sigma=5.0,
            mode='gaussian',
            downsample=1):

        self.segmentation = segmentation
        self.descriptor = descriptor
        self.mask = mask
        try:
            self.sigma = tuple(sigma)
        except:
            self.sigma = (sigma,)*3
        self.mode = mode
        self.downsample = downsample
        self.voxel_size = None
        self.context = None
        self.skip = False

        self.extractor = LsdExtractor(self.sigma, self.mode, self.downsample)

    def setup(self):

        spec = self.spec[self.segmentation].copy()
        spec.dtype = np.float32

        self.voxel_size = spec.voxel_size
        self.provides(self.descriptor, spec)

        if self.mask:
            self.provides(self.mask, spec.copy())

        if self.mode == 'gaussian':
            self.context = tuple(s*3.0 for s in self.sigma)
        elif self.mode == 'sphere':
            self.context = tuple(self.sigma)
        else:
            raise RuntimeError("Unkown mode %s"%mode)

    def prepare(self, request):

        if self.descriptor in request:

            # increase segmentation ROI to fit Gaussian
            context_roi = request[self.descriptor].roi.grow(
                self.context,
                self.context)
            grown_roi = request[self.segmentation].roi.union(context_roi)
            request[self.segmentation].roi = grown_roi

            del request[self.descriptor]
            self.skip = False

        else:

            self.skip = True

        if self.mask and self.mask in request:
            del request[self.mask]

    def process(self, batch, request):

        if self.skip:
            return

        dims = len(self.voxel_size)

        assert dims == 3, "AddLocalShapeDescriptor only works on 3D arrays."

        segmentation_array = batch.arrays[self.segmentation]

        # get voxel roi of requested descriptors -- this is the only region in
        # which we have to compute the descriptors
        seg_roi = segmentation_array.spec.roi
        descriptor_roi = request[self.descriptor].roi
        voxel_roi_in_seg = (
            seg_roi.intersect(descriptor_roi) -
            seg_roi.get_offset())/self.voxel_size

        descriptor = self.extractor.get_descriptors(
            segmentation_array.data,
            roi=voxel_roi_in_seg)

        # create descriptor array
        descriptor_spec = self.spec[self.descriptor].copy()
        descriptor_spec.roi = request[self.descriptor].roi.copy()
        descriptor_array = Array(descriptor, descriptor_spec)

        # create mask array
        if self.mask and self.mask in request:
            channel_mask = (segmentation_array.crop(descriptor_roi).data!=0).astype(np.float32)
            assert channel_mask.shape[-3:] == descriptor.shape[-3:]
            mask = np.array([channel_mask]*descriptor.shape[0])
            batch.arrays[self.mask] = Array(mask, descriptor_spec.copy())

        # crop segmentation back to original request
        seg_request_roi = request[self.segmentation].roi
        cropped_segmentation_array = segmentation_array.crop(seg_request_roi)

        batch.arrays[self.segmentation] = cropped_segmentation_array
        batch.arrays[self.descriptor] = descriptor_array
