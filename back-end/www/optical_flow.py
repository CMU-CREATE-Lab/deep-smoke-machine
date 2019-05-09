import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

class OpticalFlow(object):
    # Use this function to initialize parameters
    # Input:
    #   - input path of rgb video
    #   - output filename of the 4d rgb array
    #   - output filename of the 4d optical flow array
    #   - output file path of the arrays (eg Documents/arrays)
    #   - pixel threshold for optical flow array
    #   - pixel threshold for saturation
    #   - percentage of frame thresholded for acceptable smoke candidacy
    #   - method for computing optical flow (1 = Farneback method, 2 = TVL1 method)
    #   - number of frames to check when thresholding
    def __init__(self, rgb_vid_in_p=None, rgb_4d_out_p=None, flow_4d_out_p=None,
                 save_file_path=None, flow_threshold=255, sat_threshold=255,
                 frame_threshold=1, flow_type=1, desired_frames=1):
        self.rgb_vid_in_p = rgb_vid_in_p
        self.rgb_4d_out_p = rgb_4d_out_p
        self.flow_4d_out_p = flow_4d_out_p
        self.save_file_path = save_file_path
        self.flow_threshold = flow_threshold
        self.sat_threshold = sat_threshold
        self.frame_threshold = frame_threshold
        self.flow_type = flow_type
        self.fin_hsv_array = None
        self.thresh_4d = None
        self.rgb_4d = None
        self.rgb_filtered = None
        self.flow_4d = None
        self.desired_frames = desired_frames
        # print("constructor") TODO
        pass

    # Compute optical flow images from a batch of rgb images
    # Read rgb images and save the output hsv optical flow images to disk
    # For the output hsv image, optical direction and length are coded by hue
    # and saturation respectively Input: a 4D raw image array in rgb color
    # (rgb channel, time (in frames), height, width)
    # Output: a 4D optical flow image array in hsv color (hsv channel, time
    # (in frames), height, width)
    # IMPORTANT: you need to handle edge cases, such as only one input images
    def batch_optical_flow(self, rgb_4d=None, save_path=None):
        length, height, width = rgb_4d.shape[0], rgb_4d.shape[1], rgb_4d.shape[2]
        self.fin_hsv_array = np.zeros((length, height, width, 2))
        rgb_4d = rgb_4d.astype(int)
        rgb_4d = np.uint8(rgb_4d)
        previous_frame = rgb_4d[0, :, :, :]
        previous_gray = cv.cvtColor(previous_frame, cv.COLOR_RGB2GRAY)
        self.hsv = np.zeros_like(previous_frame)
        self.hsv[..., 1] = 0
        count = 0
        for i in rgb_4d:
            current_frame = i
            current_gray = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)
            if self.flow_type == 1:
                flow = cv.calcOpticalFlowFarneback(previous_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 2)
            elif self.flow_type == 2:
                optical_flow = cv.DualTVL1OpticalFlow_create()
                flow = optical_flow.calc(previous_gray, current_gray, None)
            magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
            # channel 0 represents direction
            self.hsv[..., 0] = angle * 180 / np.pi / 2
            # channel 2 represents magnitude
            self.hsv[..., 2] = magnitude*100
            # self.hsv[..., 2] = np.minimum(magnitude*5, 255)
            if not save_path == None:
                cv.imwrite(save_path + 'hsvframe%d.jpg' % count, self.hsv)
            self.fin_hsv_array[count, :, :, :] = self.hsv[:, :, [0, 2]]
            previous_frame = current_frame
            if not save_path == None:
                cv.imwrite(save_path + 'rgbframe%d.jpg' % count, current_frame)
            count += 1
        cv.destroyAllWindows()

        # print("batch optical flow") TODO
        return self.fin_hsv_array

    # Process an encoded video (h.264 mp4 format) into a 4D rgb image array
    # Input: path to a video clip
    # Output: a 4D image array in rgb color (rgb channel, time (in frames),
    # height, width)
    def vid_to_imgs(self, rgb_vid_path=None):
        capture = cv.VideoCapture(rgb_vid_path)
        ret, previous_frame = capture.read()

        height = np.size(previous_frame, 0)
        width = np.size(previous_frame, 1)
        length = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
        fin_rgb_array = np.zeros((length, height, width, 3))

        if length <= 1:
            return None

        for i in range(length):
            fin_rgb_array[i, :, :, :] = cv.cvtColor(previous_frame, cv.COLOR_BGR2RGB)
            ret, current_frame = capture.read()
            previous_frame = current_frame

        capture.release()
        # print("video to images") TODO
        return fin_rgb_array

    # Read a video clip and save processed image arrays to disk
    # Input:
    # - path for reading the video clip
    # - path for saving the 4D rgb image array (raw images)
    # - path for saving the 4D hsv image array (optical flow)
    def step(self, rgb_vid_in_p=None, rgb_4d_out_p=None, flow_4d_out_p=None, save_path=None):
        # print("process video from %s" % rgb_vid_in_p) TODO
        if self.rgb_vid_in_p == None:
            return None
        rgb_4d = self.vid_to_imgs(self.rgb_vid_in_p)
        self.rgb_4d = np.copy(rgb_4d)
        self.rgb_filtered = np.copy(self.rgb_4d)

        flow_4d = self.batch_optical_flow(rgb_4d, save_path)

        # Reshape rgb_4d to a pytorch readable format:
        # (Channel, time in frames, height, width)
        rgb_4d = rgb_4d.swapaxes(0, 3)
        rgb_4d = rgb_4d.swapaxes(1, 3)
        rgb_4d = rgb_4d.swapaxes(2, 3)

        # Reshape flow_4d to a pytorch readable format:
        # (Channel, time in frames, height, width)
        flow_4d = flow_4d.swapaxes(0, 3)
        flow_4d = flow_4d.swapaxes(1, 3)
        flow_4d = flow_4d.swapaxes(2, 3)
        # Save rgb_4d to path rgb_4d_out_p
        # print("save raw rgb images to %s" % rgb_4d_out_p) TODO
        # np.save(rgb_4d_out_p, rgb_4d)
        # Save flow_4d to path flow_4d_out_p
        # print("save flow hsv images to %s" % flow_4d_out_p) TODO
        # np.save(flow_4d_out_p, flow_4d)

    def threshold(self, pixel_threshold, saturation_threshold, frame_threshold, desired_frames):
        # print("determining video threshold") TODO
        if type(self.fin_hsv_array) != np.ndarray:
            return None
        num_frames = int(np.shape(self.fin_hsv_array)[0]*desired_frames)
        acceptable_per_frame = []
        self.thresh_4d = np.copy(self.fin_hsv_array)
        frame = 0
        for hsv in self.fin_hsv_array:
            bin_img = self.optical_flow_threshold(pixel_threshold, frame, hsv)
            self.saturation_threshold(saturation_threshold, acceptable_per_frame, frame, bin_img)
            frame += 1
        if desired_frames < 1:
            acceptable_per_frame.sort()
            sub_s = acceptable_per_frame[-num_frames:]
            print(sub_s)
            for i in sub_s:
                # print(i) TODO
                if i > frame_threshold:
                    print("acceptable")
                    return True
            print("no smoke detected")
            return False
        for i in acceptable_per_frame:
            if i > frame_threshold:
                # print("acceptable") TODO
                return True
        # print("no smoke detected") TODO
        return False

    def optical_flow_threshold(self, pixel_threshold, frame, hsv):
        self.thresh_4d[frame,:,:,:] = np.copy(hsv[:, :, :])
        # print(self.thresh_4d[frame, :, :, 1])
        bin_img = 1.0 * (self.thresh_4d[frame,:,:,1] > pixel_threshold)
        return bin_img

    def saturation_threshold(self, saturation_threshold, acceptable_per_frame, frame, bin_img):
        rgb = self.rgb_filtered[frame, :, :, :]
        hsv = np.copy(rgb).astype(np.uint8)
        hsv = cv.cvtColor(hsv, cv.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        saturation = 1.0 * np.where(saturation < saturation_threshold, 0, saturation)
        saturation = 1.0 * np.where(bin_img == 0, 0, saturation)
        zeros = np.zeros_like(saturation)

        bin_img = np.where(saturation == 0, 0, bin_img)

        bin_img_shape = np.shape(bin_img)
        acceptable_per_frame.append(np.sum(bin_img[:]) / (bin_img_shape[0] * bin_img_shape[1]))
        self.thresh_4d[frame, :, :, 1] = bin_img

        self.rgb_filtered[frame, :, :, 0] = zeros
        self.rgb_filtered[frame, :, :, 1] = saturation
        self.rgb_filtered[frame, :, :, 2] = zeros

    def compute_optical_flow(self):
        self.step(self.rgb_vid_in_p, self.rgb_4d_out_p, self.flow_4d_out_p, self.save_file_path)

    def contains_smoke(self):
        return self.threshold(self.flow_threshold, self.sat_threshold, self.frame_threshold, self.desired_frames)

    def show_flow(self):
        for i in range(np.shape(self.rgb_4d)[0]):
            plt.subplot(141), plt.imshow(self.thresh_4d[i,:,:,1]), plt.title('thresh')
            plt.subplot(142), plt.imshow(self.fin_hsv_array[i,:,:,1].astype(np.uint8)), plt.title('hsv')
            plt.subplot(143), plt.imshow(self.rgb_4d[i, ...].astype(np.uint8)), plt.title('rgb')
            plt.subplot(144), plt.imshow(self.rgb_filtered[i, ...].astype(np.uint8)), plt.title('filt')
            plt.figure()
            plt.draw()
            plt.pause(0.001)
