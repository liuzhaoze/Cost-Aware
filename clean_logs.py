import os
import shutil

if __name__ == "__main__":
    log_directory = "./logs"
    for item in os.listdir(log_directory):
        item_path = os.path.join(log_directory, item)
        if os.path.isdir(item_path) and item != ".placeholder":
            shutil.rmtree(item_path)
            print(f"Removed {item_path}")
