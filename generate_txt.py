import os


def generate_txt_mix(data_src_path, txt_save_path, txt_name):
    train_list = []
    val_list = []

    # 修改路径指向WiFi数据目录
    train_src = os.path.join(data_src_path, "train_data/WiFi/")
    test_src = os.path.join(data_src_path, "val_data/WiFi/")

    # 处理训练集
    for file in os.listdir(train_src):
        if not file.endswith('.npy'):  # 确保只处理npy文件
            continue
        filename = os.path.splitext(file)[0]
        parts = filename.split('_')
        if len(parts) < 3:  # 确保文件名格式正确
            print(f"跳过格式异常文件: {file}")
            continue
        # 假设文件名格式为 P1_1_0 （志愿者_动作_样本序号）
        vol_index = parts[0]  # 志愿者ID (如P1)
        act_index = parts[1]  # 动作ID (如1)
        train_list.append(f"{filename},{vol_index},{act_index}\n")

    # 处理测试集
    for file in os.listdir(test_src):
        if not file.endswith('.npy'):
            continue
        filename = os.path.splitext(file)[0]
        parts = filename.split('_')
        if len(parts) < 3:
            print(f"跳过格式异常文件: {file}")
            continue
        vol_index = parts[0]
        act_index = parts[1]
        val_list.append(f"{filename},{vol_index},{act_index}\n")

    # 写入训练文件
    train_path = os.path.join(txt_save_path, f"{txt_name}_train.txt")
    with open(train_path, 'w') as f:
        f.writelines(train_list)
        print(f"生成训练集列表: {len(train_list)} 条记录")

    # 写入验证文件
    val_path = os.path.join(txt_save_path, f"{txt_name}_val.txt")
    with open(val_path, 'w') as f:
        f.writelines(val_list)
        print(f"生成验证集列表: {len(val_list)} 条记录")


if __name__ == '__main__':
    # 配置路径参数
    root_path = "./dataset/HS-WIRF_dataset/"
    data_src_path = "./dataset/HS-WIRF_dataset/"
    txt_name = "dml"  # 修改输出文件前缀

    generate_txt_mix(
        data_src_path=data_src_path,
        txt_save_path=root_path,
        txt_name=txt_name
    )
