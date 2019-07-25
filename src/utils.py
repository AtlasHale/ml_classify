import os
import tarfile
import shutil
from google.cloud import storage
import threading


def unpack(out_dir, tar_file, learning_curve=False):
    """
    Function to instantiate data sets from tar files
    :param learning_curve: used to indicate if the tar file was made by learning_curve.py
    :param out_dir: output directory to extract tar file to
    :param tar_file: tar file name as it shows in the a cloud bucket
    :return: None, file should be extracted into out_dir
    """
    project_home = os.environ.get('PROJECT_HOME')
    if not os.path.exists(os.path.join(project_home, 'data')):
        os.mkdir(os.path.join(project_home, 'data'))
    if 'tar.gz' in tar_file and learning_curve is False:
        download_gs(tar_file=tar_file, tar_bucket=os.environ.get('TAR_BUCKET'), out_dir=os.path.join(out_dir, 'data'))
        print('Unpacking {} to {}'.format(tar_file, out_dir))
        tar = tarfile.open(os.path.join(out_dir, 'data', tar_file))
        tar.extractall(path=os.path.join(out_dir, 'data'))
        tar.close()
    elif 'tar.gz' in tar_file and learning_curve is True:
        download_gs(tar_file=tar_file, tar_bucket=os.environ.get('TAR_BUCKET'), out_dir=os.path.join(out_dir, 'data'))
        print(f'downloading tar file: {tar_file}\nFrom bucket: {os.environ.get("TAR_BUCKET")}\nTo directory: {os.path.join(out_dir, "data")}')
        def threaded_extract(tar_file):
            print('Unpacking in a thread {} to {}'.format(tar_file, out_dir))
            tar = tarfile.open(os.path.join(out_dir, 'data', tar_file))
            tar.extractall(path=os.path.join(out_dir, 'data'))
            tar.close()
            print('finished unpacking thread')
        thread1 = threading.Thread(target=threaded_extract(tar_file))
        thread1.start()
        thread1.join()
        print(os.listdir(os.path.join(out_dir, 'data')))
        if 'train' in tar_file and 'temp' in os.listdir(os.path.join(out_dir, 'data')):
            shutil.rmtree(os.path.join(out_dir, 'data', 'train'))
            os.rename(os.path.join(out_dir, 'data', 'temp'), os.path.join(out_dir, 'data', 'train'))
    elif os.path.isfile(tar_file) and 'tar.gz' in tar_file and 's3' not in tar_file:
        print('Unpacking {}'.format(tar_file))
        tar = tarfile.open(tar_file)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        tar.extractall(path=out_dir)
        tar.close()
    elif 'tar.gz' in tar_file and 's3' in tar_file:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # download first then untar
        t = download_s3(tar_file, out_dir)
        print('Unpacking {}'.format(t))
        tar = tarfile.open(t)
        tar.extractall(path=out_dir)
        tar.close()


def has_number(tar_file):
    return any(letter.isdigit() for letter in tar_file)


def download_gs(tar_file, tar_bucket, out_dir):
    """
    Downloading from google storage buckets
    :param tar_file: tar file name in bucket. If location is gs://mbari-bucket/train.tar.gz, tar_file = 'train.tar.gz'
    :param tar_bucket: tar bucket to look in. If location is gs://mbari-bucket/train.tar.gz, tar_bucket = 'mbari-bucket'
    :param out_dir: output directory, usually something like a /data folder.
    :return: None, tar file should be downloaded from cloud to local.
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(tar_bucket)
    blob = bucket.blob(tar_file)
    out_file = os.path.join(out_dir, tar_file)
    blob.download_to_filename(out_file)



def list_bucket_contents(bucket):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket)
    classes = []
    blobs = bucket.list_blobs()
    uniques = set()
    for item in blobs:
        l = str(item).split(', ')
        current_species = l[1].split('/')[1]
        if current_species not in uniques:
            classes.append(current_species)
            uniques.add(current_species)
    return classes


def download_s3(source_file, target_dir):
    try:
        import boto3
        from botocore.client import Config
        import botocore
        import os
        env = os.environ.copy()
        from urllib.parse import urlparse
        urlp = urlparse(source_file)
        endpoint_url = 'http://' + urlp.netloc
        bucket_name = urlp.path.split('/')[1]
        KEY_IN = urlp.path.split(bucket_name + '/')[1]
        KEY_OUT = os.path.join(target_dir, os.path.basename(urlp.path))
        print('Downloading {} from {} to {}'.format(KEY_IN, endpoint_url, KEY_OUT))
        s3 = boto3.resource('s3',
                            endpoint_url=endpoint_url,
                            aws_access_key_id=env['AWS_ACCESS_KEY_ID'],
                            aws_secret_access_key=env['AWS_SECRET_ACCESS_KEY'],
                            config=Config(signature_version='s3v4'),
                            region_name='us-east-1')

        try:
            s3.Bucket(bucket_name).download_file(KEY_IN, KEY_OUT)
            return KEY_OUT
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            print(e)
    except Exception as e:
        raise(e)

