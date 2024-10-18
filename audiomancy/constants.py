import os

ABSOLUTE_PATH = os.path.join(os.path.dirname(__file__))

DATASETS_PATH = os.path.join(ABSOLUTE_PATH,'common','datasets')
STEMS_PATH = os.path.join(ABSOLUTE_PATH,'common','stems')
TEMP_PATH = os.path.join(ABSOLUTE_PATH,'common','temp')

JAMS_FILE_200 = os.path.join(ABSOLUTE_PATH,'common','200_soundscapes.jams')
JAMS_FILE_1000 = os.path.join(ABSOLUTE_PATH,'common','1000_soundscapes.jams')
STEM_METADATA = os.path.join(ABSOLUTE_PATH,'common','metadata.json')

LABELS = {
    'accoustic':0,
    'bass':1,
    'drums':2,
    'piano':3,
    'strings':4,
    'vocals':5
}