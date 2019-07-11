import os
import tarfile
import shutil


def unpack(out_dir, tar_file):

    if os.path.isfile(tar_file) and 'tar.gz' in tar_file and 's3' not in tar_file:
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
    elif 'tar.gz' not in tar_file:
        project_home = os.environ.get('PROJECT_HOME')
        shutil.copytree(project_home+'/data/'+tar_file, out_dir)
    # create labels from directory names
    

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

