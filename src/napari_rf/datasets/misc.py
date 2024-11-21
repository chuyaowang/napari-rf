import os, re

def delete_empty_folders(root):
    deleted = set()

    for current_dir, subdirs, files in os.walk(root, topdown=False):

        still_has_subdirs = False
        for subdir in subdirs:
            if os.path.join(current_dir, subdir) not in deleted:
                still_has_subdirs = True
                break

        if not any(files) and not still_has_subdirs:
            os.rmdir(current_dir)
            deleted.add(current_dir)

    return deleted



if __name__ == '__main__':
    root = '/media/philipp/seagate5tb/hydra'

    old_paths = []
    p,c,t = [],[],[]
    for path, _, files in os.walk(root):
        for file in files:
            if file.endswith('.tif'):
                old_path = f"{path}/{file}"
                old_paths.append(old_path)
                p.append(re.findall('position_(\d+)',old_path))
                c.append(re.findall('channel_(\d+)',old_path))
                t.append(re.findall('frame_(\d+)',old_path))
    p_fill = max([len(str(int(match[0]))) for match in p if match])
    c_fill = max([len(str(int(match[0]))) for match in c if match])
    t_fill = max([len(str(int(match[0]))) for match in t if match])

    for old_path in old_paths:
        p = str(int(re.findall('position_(\d+)', old_path)[0]))
        c = str(int(re.findall('channel_(\d+)', old_path)[0]))
        t = str(int(re.findall('frame_(\d+)', old_path)[0]))

        new_dir = f"{root}/dataset/position_{p.zfill(p_fill)}/channel_{c.zfill(c_fill)}/"
        os.makedirs(new_dir, exist_ok=True)
        new_path = f"{new_dir}/frame_{t.zfill(t_fill)}.tif"

        os.replace(old_path, new_path)

    delete_empty_folders(root)

