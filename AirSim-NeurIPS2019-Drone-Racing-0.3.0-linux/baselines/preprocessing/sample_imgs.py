import os
from os.path import isfile, join

def deletefile(pathIn):
    SAMPLE_FREQ = 10 # sample 1 image for every 1 image
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

    # for sorting the file names properly
    # files.sort(key=lambda x: int(x[-9:-4]))

    for i in range(len(files)):
        filename = pathIn + files[i]
        no_file = int(files[i][1:-4])	# use this to get the number in the name
					# e.g. name = 'Q12345.jpg' => name[1:-4] to get '12345'
        if(no_file % SAMPLE_FREQ != 0):
            os.remove(filename)
            print("delete file", filename)


def main():
    pathIn = '/home/jedsadakorn/AirSim_Qualification/Q1000/'
    deletefile(pathIn)


if __name__ == "__main__":
    main()

