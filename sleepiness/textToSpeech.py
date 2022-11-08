"""
This script was used to create our mp3 files for our different alerts
"""
from gtts import gTTS
import os 
  
localPath = os.path.dirname(os.path.realpath(__file__))
localPath = localPath.replace('\\', '/')
localPath = str(localPath.replace('c:/', 'C:/')) + "/"

# The text that you want to convert to audio 
mytext = 'ATTENTION. Drowsiness detected because you are yawning. Please pull the car over'
  
# Language in which you want to convert 
language = 'en'
  
# Passing the text and language to the engine,  
# here we have marked slow=False. Which tells  
# the module that the converted audio should  
# have a high speed 
myobj = gTTS(text=mytext, lang=language, slow=False) 
  
# Saving the converted audio in a mp3 file
myobj.save(localPath + "/sounds/alarmMouth.mp3")
