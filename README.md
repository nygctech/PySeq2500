# PySeq2500V2
Control HiSeq2500 V2 with instruments as objects

#Example
import pyseq
hs = pyseq.HiSeq()                  # Create HiSeq Object
hs.initializeCams()                 # Initialize Cameras
hs.initializeInstruments()          # Initialize x,y,z, & objective stages. Initialize lasers and optics (filters)

hs.1l.set_power(100)                # Set green laser power to 100 mW
hs.12.set_power(100)                # Set red laser power to 100 mW
hs.y.move(-180000)                  # Move stage to top right corner of Flow Cell A
hs.x.move(17571)
hs.z.move([21250, 21250, 21250])    # Raise z stage
hs.obj.move(30000)                  # Move objective to middle-ish
hs.move_ex(1,'open')                # Move excitation filter 1 to open position
hs.move_ex(2,'open')                # Move excitation filter 2 to open position
hs.l1.get_power()                   # Get green laser power (mW i think)
hs.l2.get_power()                   # Get red laser power   (mW i think)


# Set where you want to save the images
hs.image_path = 'C:\\Users\\Public\\Documents\\PySeq2500\\Images\\'

# Take an image
# take_picture(# frames, bundle height, image_name (optional, defaults to timestamp)
hs.take_picture(32, 128) 
