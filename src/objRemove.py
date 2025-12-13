
import copy
import cv2
import numpy as np
import torchvision.transforms as T  
from torchvision.io import read_image

class ObjectRemove():

    def __init__(self, segmentModel = None, rcnn_transforms = None, inpaintModel= None, image_path = '') -> None:
        self.segmentModel = segmentModel
        self.inpaintModel = inpaintModel
        self.rcnn_transforms = rcnn_transforms
        self.image_path = image_path
        self.highest_prob_mask = None
        self.image_orig = None
        self.image_masked = None
        self.box = None


    def run(self):
        # Step 1: Read in and preprocess the image
        images = self.read_and_preprocess_image()

        # Step 2: Perform segmentation
        out = self.perform_segmentation(images)

        # Step 3: User interaction: Clicking
        ref_points = self.click_me()

        # Step 4: Generate mask based on user input
        mask = self.generate_mask(images,out, ref_points)

        # Step 5: Threshold the mask
        thresholded_mask = self.threshold_mask(mask)

        # Step 6: Mask the original image
        masked_image = self.mask_original_image(images[0], thresholded_mask)

        # Step 7: Inpainting
        output = self.perform_inpainting(masked_image)
        
        # Return the final inpainted image
        return output

    # Separate methods for each step
    def read_and_preprocess_image(self):
        print('Reading in image...')
        images = self.preprocess_image()
        self.image_orig = images
        return images

    def perform_segmentation(self, images):
        print("Segmentation...")
        output = self.segment(images)
        return output[0]

    def click_me(self):
        print('User click...')
        ref_points = self.user_click()
        self.box = ref_points
        return ref_points

    def generate_mask(self, images,out, ref_points):
        print('Generating mask...')
        mask = self.find_mask(out, ref_points)
        mask[mask > 0.1] = 1
        mask[mask < 0.1] = 0
        self.highest_prob_mask = mask
        self.image_masked = images[0] * (1 - mask)
        return mask

    def threshold_mask(self, mask):
        print('Thresholding mask...')
        return mask

    def mask_original_image(self, original_image, mask):
        print('Masking original image...')
        return original_image

    def perform_inpainting(self, masked_image):
        print('Inpainting...')
        output = self.inpaint()
        return output


    def percent_within(self, nonzeros, rectangle):
        rect_ul = rectangle[0]
        rect_br = rectangle[1]

        inside_count = 0

        for _, y, x in nonzeros:
            if x >= rect_ul[0]:
                if x <= rect_br[0]:
                    if y >= rect_ul[1]:
                        if y <= rect_br[1]:
                            inside_count += 1


        total_points = len(nonzeros)

        if total_points == 0:
            return 0

        return inside_count / total_points




    def iou(self, boxes_a, boxes_b):
        x1 = np.maximum(boxes_a[:, 0], boxes_b[:, 0])
        y1 = np.maximum(boxes_a[:, 1], boxes_b[:, 1])
        x2 = np.minimum(boxes_a[:, 2], boxes_b[:, 2])
        y2 = np.minimum(boxes_a[:, 3], boxes_b[:, 3])

        width = x2 - x1
        height = y2 - y1
        width[width < 0] = 0
        height[height < 0] = 0

        intersection_area = width * height
        area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
        union_area = area_a + area_b - intersection_area

        iou_scores = intersection_area / (union_area + 0.00001)
        return iou_scores


    def find_mask(self, rcnn_output, rectangle):
        bounding_boxes= rcnn_output['boxes'].detach().numpy()
        masks = rcnn_output['masks']

        ref_boxes  = np.array([rectangle], dtype=object)
        ref_boxes = np.repeat(ref_boxes, bounding_boxes.shape[0], axis=0)

        ious= self.iou(ref_boxes, bounding_boxes)

        best_ind = np.argmax(ious)

        return masks[best_ind]


    def preprocess_image(self):
        img= [read_image(self.image_path)]
        _,h,w = img[0].shape
        size = min(h,w)
        if size > 512:
            img[0] = T.Resize(512, max_size=680, antialias=True)(img[0])

        images_transformed = [self.rcnn_transforms(d) for d in img]
        return images_transformed
   
    
    def segment(self,images):
        out = self.segmentModel(images)
        return out

    def user_click(self):
        ref_point = []
        cache=None
        draw = False


        def click(event, x, y, flags, param):
            nonlocal ref_point,cache,img, draw
            if event == cv2.EVENT_LBUTTONDOWN:
                draw = True
                ref_point = [x, y]
                cache = copy.deepcopy(img)

            elif event == cv2.EVENT_MOUSEMOVE:
                if draw:
                    img = copy.deepcopy(cache)
                    cv2.rectangle(img, (ref_point[0], ref_point[1]), (x,y), (0, 0, 255), 2)
                    cv2.imshow('image',img)


            elif event == cv2.EVENT_LBUTTONUP:
                draw = False
                ref_point += [x,y]
                ref_point.append((x, y))
                cv2.rectangle(img, (ref_point[0], ref_point[1]), (ref_point[2], ref_point[3]), (0, 0, 255), 2)
                cv2.imshow("image", img)


        img = self.image_orig[0].permute(1,2,0).numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        clone = img.copy()

        cv2.namedWindow("image")

        cv2.setMouseCallback('image', click)

        while True:
            cv2.imshow("image", img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("r"):
                img = clone.copy()
            
            elif key == 13:  # Enter key
                break
        cv2.destroyAllWindows()

        
        return ref_point
    
    def inpaint(self):
        output = self.inpaintModel.infer(self.image_orig[0], self.highest_prob_mask, return_vals=['inpainted'])
        return output[0]

