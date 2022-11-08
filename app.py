"""
Solution inspired by and built upon the code of Adrian Rosebrock:
Retrieved March 18, 2020, from https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/

The pretrained landmark detection model is based on the work of:
Sagonas, C., Antonakos, E., Tzimiropoulos, G., Zafeiriou, S., & Pantic, M. (2016).
300 faces in-the-wild challenge: Database and results. Image and vision computing, 47, 3-18.
"""
from sleepiness import sleepiness_app


def app():
    sleepiness_app(debug_mode=True, device_id=0)


if __name__ == "__main__":
    app()
