import os

def get_last_six_characters(file_name):
    base_name = os.path.basename(file_name)  # 获取文件名
    name, extension = os.path.splitext(base_name)  # 分离文件名和扩展名
    return int(name[-6:])  # 返回最后6位字符

def main():
    directory = r'/media/lin/DATASETS/human36m/images/s_01_act_15_subact_01_ca_03'  # 指定目录
    b = 0
    ma = 0
    for filename in os.listdir(directory):
        b += 1
        if os.path.isfile(os.path.join(directory, filename)):
            a = get_last_six_characters(filename)
            ma = max(a,ma)
    print(ma)
    print(b)

if __name__ == "__main__":
    main()