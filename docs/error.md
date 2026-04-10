# Upload file dataset domain (train.jsonl, validation.jsonl, test.jsonl)
# File ada di: D:\2-Project\AQG\dataset_aqg\output_domain\

from google.colab import files

print('Upload file dari: dataset_aqg/output_domain/')
print('Pilih 3 file: train.jsonl, validation.jsonl, test.jsonl')
uploaded = files.upload()

# Pindahkan ke direktori yang benar
for filename in uploaded.keys():
    os.rename(filename, f'/content/dataset_aqg/output_domain/{filename}')
    print(f'✓ {filename} → /content/dataset_aqg/output_domain/')


Upload file dari: dataset_aqg/output_domain/
Pilih 3 file: train.jsonl, validation.jsonl, test.jsonl

---------------------------------------------------------------------------
KeyboardInterrupt                         Traceback (most recent call last)
/tmp/ipykernel_6062/2782464141.py in <cell line: 0>()
      6 print('Upload file dari: dataset_aqg/output_domain/')
      7 print('Pilih 3 file: train.jsonl, validation.jsonl, test.jsonl')
----> 8 uploaded = files.upload()
      9 
     10 # Pindahkan ke direktori yang benar

/usr/local/lib/python3.12/dist-packages/google/colab/files.py in upload(target_dir)
     67   """
     68 
---> 69   uploaded_files = _upload_files(multiple=True)
     70   # Mapping from original filename to filename as saved locally.
     71   local_filenames = dict()

/usr/local/lib/python3.12/dist-packages/google/colab/files.py in _upload_files(multiple)
    159 
    160   # First result is always an indication that the file picker has completed.
--> 161   result = _output.eval_js(
    162       'google.colab._files._uploadFiles("{input_id}", "{output_id}")'.format(
    163           input_id=input_id, output_id=output_id

/usr/local/lib/python3.12/dist-packages/google/colab/output/_js.py in eval_js(script, ignore_result, timeout_sec)
     38   if ignore_result:
     39     return
---> 40   return _message.read_reply_from_input(request_id, timeout_sec)
     41 
     42 

/usr/local/lib/python3.12/dist-packages/google/colab/_message.py in read_reply_from_input(message_id, timeout_sec)
     94     reply = _read_next_input_message()
     95     if reply == _NOT_READY or not isinstance(reply, dict):
---> 96       time.sleep(0.025)
     97       continue
     98     if (

KeyboardInterrupt: