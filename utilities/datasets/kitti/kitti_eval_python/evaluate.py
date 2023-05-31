import utilities.datasets.kitti.kitti_common as kitti
from utilities.datasets.kitti.eval import get_coco_eval_result, get_official_eval_result

#import argparse
#import kitti_common as kitti
#from eval import get_coco_eval_result, get_official_eval_result



def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def evaluate(label_path, result_path, label_split_file, current_class=0, coco=False, score_thresh=-1):
    dt_annos = kitti.get_label_annos(result_path)
    if score_thresh > 0:
        dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    val_image_ids = _read_imageset_file(label_split_file)
    gt_annos = kitti.get_label_annos(label_path, val_image_ids)
    if coco:
        return get_coco_eval_result(gt_annos, dt_annos, current_class)
    else:
        return get_official_eval_result(gt_annos, dt_annos, current_class)


if __name__ == '_main_':
    '''
    parser = argparse.ArgumentParser(description='KITTI Evaluation')
    parser.add_argument('--label_path', type=str, help='Path to the GT label folder')
    parser.add_argument('--result_path', type=str, help='Path to the result folder')
    parser.add_argument('--label_split_file', type=str, help='Path to the label split file (e.g., val.txt)')
    parser.add_argument('--current_class', type=int, default=0, help='Current class')
    parser.add_argument('--coco', type=bool, default=False, help='Use COCO evaluation')
    parser.add_argument('--score_thresh', type=float, default=-1, help='Score threshold')
    args = parser.parse_args()
    '''

    label_path = '/Users/strom/Desktop/monodle/data/KITTI/object/training/label_2'
    result_path = '/Users/strom/Desktop/monodle/outputs/data'
    label_split_file = '/Users/strom/Desktop/monodle/data/KITTI/ImageSets/val.txt'
    current_class = 0
    coco = False
    print('test')
    eval = evaluate(label_path, result_path, label_split_file,
             current_class, coco)
    print('test')

    for element in eval:
     print(element)
