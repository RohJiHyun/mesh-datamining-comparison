import os 



def check_allowed_ext(path, allowed_ext = []):
    filename = os.path.basename(path)
    _, ext = os.path.splitext(filename)
    if ext in allowed_ext:
        return True
    return False


def to_abs(path):
    """
        return abs path
    """
    if os.path.isabs(path):
        return path
    return os.path.abspath(path)


def exchange_root_path(target_path, from_root, to_root):
    target  = to_abs(target_path)
    from_root = to_abs(from_root)
    to_root = to_abs(to_root)

    # if 
    # target = '/book/html/wa/foo/bar/'
    # to_root = '/book/html'
    # 'wa/foo/bar'
    tmp_target = os.path.relpath(target, from_root)
    return os.path.join(to_root, tmp_target)
    