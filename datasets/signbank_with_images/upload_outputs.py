import os
import os.path

import boto3
import tqdm

ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
SECRET_KEY = os.getenv('AWS_SECRET_KEY')
BUCKET_NAME = 'signwriting-images'


def upload_file(client, filepath: str) -> str:
    with open(filepath, 'rb') as f:
        filename = os.path.basename(filepath)
        object_key = os.path.join('dataset-outputs', filename)

        client.put_object(
            Bucket=BUCKET_NAME,
            Key=object_key,
            Body=f
        )

        return client.generate_presigned_url(
            ClientMethod='get_object',
            Params={
                'Bucket': BUCKET_NAME,
                'Key': object_key,
            }
        )


def main():
    output_path = './output'
    client = boto3.client(
        's3',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY
    )

    with tqdm.tqdm(os.listdir(output_path)) as counter:
        for file in os.listdir(output_path):
            upload_file(client, os.path.join(output_path, file))
            counter.update(1)


if __name__ == '__main__':
    main()
