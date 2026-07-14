import os
import re


def filter_and_delete_images(folder_path):
    # 1. 支持的常见图像格式
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    # 2. 获取文件夹下所有图片路径
    all_files = os.listdir(folder_path)
    image_files = [f for f in all_files if f.lower().endswith(valid_extensions)]

    # 3. 关键步骤：自然排序（Natural Sort）
    # 避免出现 [1.jpg, 10.jpg, 2.jpg] 这种系统默认的混乱排序
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    image_files.sort(key=natural_sort_key)

    print(f"共检测到 {len(image_files)} 张图片，开始执行「保留1张，删除4张」...")

    kept_count = 0
    deleted_count = 0

    # 4. 遍历并处理
    for index, file_name in enumerate(image_files):
        full_path = os.path.join(folder_path, file_name)

        # 每 5 张为一组，索引为 0, 5, 10... 的保留，其余删除
        if index % 5 == 0:
            kept_count += 1
            print(f"[保留] -> {file_name}")

            # 如果你确实需要用 OpenCV 处理留下来的这张图，写在这里：
            # import cv2
            # img = cv2.imread(full_path)
            # # 你的 OpenCV 处理逻辑...

        else:
            try:
                os.remove(full_path)
                deleted_count += 1
            except OSError as e:
                print(f"删除失败 {file_name}: {e}")

    print("\n--- 处理完成 ---")
    print(f"成功保留: {kept_count} 张")
    print(f"成功删除: {deleted_count} 张")


# 使用时，把这里的路径替换成你存放图片的文件夹路径
folder_path = "./batch_input"
filter_and_delete_images(folder_path)