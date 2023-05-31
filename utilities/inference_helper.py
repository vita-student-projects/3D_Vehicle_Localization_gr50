import os
import tqdm
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
import random
import time


import cv2
from utilities.datasets.kitti.kitti_utils import Calibration, Object3d
from utilities.datasets.utils import draw_projected_box3d, draw_projected_box3d_2
import matplotlib.pyplot as plt

from utilities.save_helper import load_checkpoint
from utilities.decode_helper import extract_dets_from_outputs
from utilities.decode_helper import decode_detections



class Inference(object):
    def __init__(self, cfg, model, dataloader, logger, eval=False, number_inference=0, scene=False):
        self.number_inference = number_inference
        self.scene = scene
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.max_objs = dataloader.dataset.max_objs    # max objects per images, defined in dataset
        self.class_name = dataloader.dataset.class_name
        self.output_dir = './outputs'
        self.dataset_type = cfg.get('type', 'KITTI')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.eval = eval
        self.id_imgs = []
        self.calibs_imgs = []



    def test(self):
        assert self.cfg['mode'] in ['single']

        # test a single checkpoint
        if self.cfg['mode'] == 'single':
            assert os.path.exists(self.cfg['checkpoint'])
            load_checkpoint(model=self.model,
                            optimizer=None,
                            map_location=self.device,
                            logger=self.logger,
                            path = self.cfg['checkpoint'])
            self.model.to(self.device)
            self.inference()
            if self.scene:
                self.plot_inference_scene()
            else:
                self.plot_inference()




    def inference(self):
        torch.set_grad_enabled(False)
        self.model.eval()
        results = {}
        progress_bar = tqdm.tqdm(total=self.number_inference, leave=True, desc='Evaluation Progress')
        # Start the timer
        start_time = time.time()
        for batch_idx, (inputs, target, info) in enumerate(self.dataloader):
            if batch_idx < self.number_inference:
                # load evaluation data and move data to GPU.
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)

                dets = extract_dets_from_outputs(outputs=outputs, K=self.max_objs)
                dets = dets.detach().cpu().numpy()
        
                # get corresponding calibs & transform tensor to numpy
                calibs = [self.dataloader.dataset.get_calib(index)  for index in info['img_id']]
                info = {key: val.detach().cpu().numpy() for key, val in info.items()}
                cls_mean_size = self.dataloader.dataset.cls_mean_size
                #decode the detection results
                dets = decode_detections(dets=dets,
                                        info=info,
                                        calibs=calibs,
                                        cls_mean_size=cls_mean_size,
                                        threshold=self.cfg.get('threshold', 0.2))
                self.id_imgs.append(info['img_id'])
                self.calibs_imgs.append(calibs)
                results.update(dets)
                progress_bar.update()
            else:
                
                break
        # End the timer
        end_time = time.time()
        elapsed_time = end_time - start_time
        # Print the elapsed time
        print(f"Elapsed time for {self.number_inference} inferences: {elapsed_time} seconds")

        progress_bar.close()

        # save the result for evaluation.
        self.logger.info('==> Saving ...')
        self.save_results(results)



    def save_results(self, results, output_dir='./outputs'):
        output_dir = os.path.join(output_dir, 'data_inference')
        os.makedirs(output_dir, exist_ok=True)

        for img_id in results.keys():
            if self.dataset_type == 'KITTI':
                output_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
            else:
                os.makedirs(os.path.join(output_dir, self.dataloader.dataset.get_sensor_modality(img_id)), exist_ok=True)
                output_path = os.path.join(output_dir,
                                           self.dataloader.dataset.get_sensor_modality(img_id),
                                           self.dataloader.dataset.get_sample_token(img_id) + '.txt')

            f = open(output_path, 'w')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()


    def plot_inference(self):
        for i in range(self.number_inference):

            fig, axs = plt.subplots(1, 2, figsize=(12, 4))
            for ax in axs.flatten():
                ax.axis('off')

            prediction_path = "/Users/strom/Desktop/monodle/outputs/data_inference/00{:04d}".format(self.id_imgs[i][0])+ '.txt'  #change absolute path to yours
            image_path = "/Users/strom/Desktop/monodle/data/KITTI/object/testing/image_2/00{:04d}".format(self.id_imgs[i][0])  +".png" 

            image = plt.imread(image_path)
            axs[0].imshow(image)
            axs[0].set_title('Original Image')

            image2 = image
            calib_3D = self.calibs_imgs[i][0]

            with open(prediction_path, 'r') as file:
                for line in file:
                    line = line.strip()  # Remove leading/trailing whitespace and newline characters
                    # Process each line here
                    test_obj = Object3d(line)
                    _ , boxes_corner = calib_3D.corners3d_to_img_boxes(np.resize(test_obj.generate_corners3d(),(1,8,3)))
                    boxes = np.resize(boxes_corner,(8,2))[::-1]
                    image2 = draw_projected_box3d(image, boxes, color=(255, 0,0), thickness=2) # Use (255, 0, 0) for red color

            # Plot image with drawn boxes
            axs[1].imshow(image2)
            axs[1].set_title('Image with Drawn Boxes')


    def plot_inference_scene(self):
        for i in range(self.number_inference):
            prediction_path = "/Users/strom/Desktop/monodle/outputs/data_inference/00{:04d}".format(self.id_imgs[i][0]) + '.txt' #change absolute path to yours
            image_path = "/Users/strom/Desktop/monodle/data/DLAV/object/testing/scene1/scene1_0{:04d}".format(self.id_imgs[i][0]) + ".png"

            image = mpimg.imread(image_path)
            image = image[50:50+384, 0:1280]

            image2 = image.copy()
            calib_3D = self.calibs_imgs[i][0]

            with open(prediction_path, 'r') as file:
                for line in file:
                    line = line.strip()  # Remove leading/trailing whitespace and newline characters
                    # Process each line here
                    test_obj = Object3d(line)
                    _, boxes_corner = calib_3D.corners3d_to_img_boxes(np.resize(test_obj.generate_corners3d(), (1, 8, 3)))
                    boxes = np.resize(boxes_corner, (8, 2))[::-1]
                    image2 = draw_projected_box3d(image, boxes, color=(255, 0, 0), thickness=2)  # Use (255, 0, 0) for red color

            output_path = f"/Users/strom/Desktop/monodle/outputs/scene1/{i}.png"
            pil_image = Image.fromarray((image2 * 255).astype(np.uint8))  # Convert the image to a PIL Image object
            pil_image.save(output_path)

                    