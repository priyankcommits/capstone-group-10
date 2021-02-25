import os, boto3
import zipfile
from pathlib import Path

def get_boto_session():
    session = boto3.Session(
        aws_access_key_id=os.environ["AWS_KEY"],
        aws_secret_access_key=os.environ["AWS_SECRET"],
        region_name="us-east-1"
    )
    return session

def get_s3_object(bucket_name, path, file_name):
    session = get_boto_session()
    s3 = session.client("s3")
    print("downloading ==========================")
    file = s3.download_file(bucket_name, path, file_name)
    return file

def put_s3_object(bucket_name, path, file_name):
    session = get_boto_session()
    s3 = session.client("s3")
    print("uploading ==========================")
    file = s3.upload_file(path, bucket_name, file_name)
    return file

def get_s3_list(bucket_name, dir_path):
    session = get_boto_session()
    s3 = session.client("s3")
    contents = s3.list_objects(Bucket=bucket_name, Prefix=dir_path)["Contents"]
    return contents

def mandatory_data():
    train_data_dir = "car_data/car_data/train/"
    test_data_dir = "car_data/car_data/test/"
    required_train_dir = Path(train_data_dir)
    required_test_dir = Path(test_data_dir)
    if not required_train_dir.is_dir() or not required_test_dir.is_dir():
        get_s3_object("capstonegroup10", "car_data.zip", "car_data.zip")
        with zipfile.ZipFile("car_data.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
    else:
        print("exists locally, not downloading")

def install_and_import(package):
    # import importlib
    # import pip
    # import pdb;pdb.set_trace()
    # try:
    #     importlib.import_module(package)
    # except ImportError:
    #     import pip
    #     pip.main(['install', package])
    # finally:
    #     globals()["tf"] = importlib.import_module(package)
    import subprocess
    subprocess.call(['pip', 'install', package])
