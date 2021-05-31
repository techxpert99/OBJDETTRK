from os import chdir
from subprocess import call,PIPE

print('Installing Requirements. Please Wait.')
chdir('/content/drive/MyDrive/AI/OBJECT_DETECTOR_TRACKER/')
call(['pip','install','-r','requirements.txt'],stdin=PIPE,stdout=PIPE,stderr=PIPE)
print('Installation Complete.')
