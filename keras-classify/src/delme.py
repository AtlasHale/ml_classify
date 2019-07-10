#!/usr/bin/env/python
import boto3
from botocore.client import Config


s3 = boto3.resource('s3',
                    endpoint_url='http://minioserver:9000',
                    aws_access_key_id='AKEXAMPLE9F123',
                    aws_secret_access_key='wJad56Utn4EMI/KDMNF/FOOBAR9877',
                    config=Config(signature_version='s3v4'),
                    region_name='us-east-1')


s3.Bucket('kerasclassifier-test').download_file('catsdogstrain.tar.gz', '/tmp/tar.gz')
