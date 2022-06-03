import sys
from argparse import ArgumentParser

from pathlib import Path
import cv2
from tqdm import tqdm


def convert_video_to_frames(vid_path, output_dir):
    """
    Converts a given video into a folder containing each frame

    :param vid_path: Path to the video file
    :param output_dir: Directory in which video frames are saved
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    
    frame_count = 0
    cap = cv2.VideoCapture(str(vid_path))
    with tqdm(desc='Converting video', unit='frames', total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if ret:
                cv2.imwrite(str(output_dir / f'frame_{frame_count:05d}.png'), frame)
                pbar.update()
                frame_count += 1
            else:
                break
    cap.release()


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = ArgumentParser(description='Converts video to folder of frames')
    parser.add_argument('-f', '--video_file', dest='video_path', required=True, help='Path to video file that needs to be converted')
    parser.add_argument('-o', '--output_directory', dest='output_dir', default='./video_output', help='Directory to output frames to')
    args, unknown = parser.parse_known_args(args)
    
    if unknown:
        print('Unknown input arguments:')
        print(unknown)
        return
    
    video_path, output_dir = Path(args.video_path), Path(args.output_dir)
    convert_video_to_frames(video_path, output_dir)
    

if __name__ == '__main__':
    main()
