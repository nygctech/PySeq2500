import logging
import warnings
import time
import os
from os.path import join


def setup_logger():
    """Create a logger and return the handle."""


    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(10)

    # Create console handler
    c_handler = logging.StreamHandler()
    c_handler.setLevel(21)
    # Create file handler
    f_handler = logging.FileHandler(hs.log_path)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(asctime)s - %(message)s', datefmt = '%Y-%m-%d %H:%M')
    f_format = logging.Formatter('%(asctime)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


# Test LEDS
def test_led():
    try:
        logger.log(21, 'Testing LEDs')
        hs.f.initialize()
        for i in [1,2]:
          for color in hs.f.led_dict.keys():
              hs.f.LED(i, color)
          time.sleep(2)
        logger.log(21, 'LEDs Nominal')
        status = True
    except:
        logger.log(21, 'Check COM port')
        logger.log(21, 'LEDs Failed')
        status = False

    return status


# Test X Stage
def test_x_stage():
    try:
        self.y.command('OFF')
        logger.log(21, 'Testing X Stage')
        hs.x.initialize()
        hs.x.move(hs.x.min_x)
        hs.x.move(hs.x.max_x)
        hs.x.move(hs.x.home)
        logger.log(21, 'X Stage Nominal')
        status = True
    except:
        status = False
        logger.log(21, 'Check COM port')
        logger.log(21, 'X Stage Failed')

    return status

# Test Y Stage
def test_y_stage():
    try:
        logger.log(21, 'Testing Y Stage')
        hs.y.initialize()
        hs.y.move(hs.y.min_y)
        hs.y.move(hs.y.max_y)
        hs.y.move(hs.y.home)
        if instrument_status['FPGA'] and hs.x.check_position(hs.x.home):
            attempts = 0
            if abs(hs.f.read_position() - hs.y.read_position()) > 10:
                hs.reset_stage()
                attempts += 1
                if attempts >= 10:
                    warnings.warn('Unable to sync FPGA & Y Stage', RuntimeWarning)
            else:
                status = True
                logger.log(21, 'Y Stage Nominal')
    except:
        status = False
        logger.log(21, 'Check COM port')
        logger.log(21, 'Y Stage Failed')

    return status

# Test Z Stage
def test_z_stage():
    z_pass = [False, False, False]
    try:
        logger.log(21, 'Testing Z Stage')
        hs.z.initialize()
        zpos = hs.z.focus_pos
        zpos_list = hs.z.move([zpos, zpos, zpos])
        for i, z in enumerate(zpos_list):
            if abs(z-zpos) <= 5:
                z_pass[i] = True
        zpos_list = hs.z.move([0, 0, 0])
        for i, z in enumerate(zpos_list):
            if z_pass[i] and abs(z-zpos) <= 5:
                z_pass[i] = True
        if all(z_pass):
            status = True
            logger.log(21, 'Z Stage Nominal')
        else:
            for i, z in enumerate(z_pass):
                if not z:
                    warnings.warn('Z Tilt Motor '+str(i)+' Failed.', RuntimeWarning)
    except:
        status = False
        logger.log(21, 'Z Stage Failed')

    return status


# Test Objective Stage
def test_objective_stage():
    try:
        logger.log(21, 'Testing Objective Stage')
        hs.obj.initialize()
        hs.obj.move(hs.obj.min_z)
        hs.obj.move(hs.obj.max_z)
        hs.obj.set_velocity(hs.obj.min_v)
        hs.obj.move(hs.obj.min_z)
        hs.obj.move(hs.obj.max_z)
        hs.obj.set_velocity(hs.obj.max_v)
        logger.log(21, 'Objective Stage Nominal')
        status = True
    except:
        status = False
        logger.log(21, 'Objective Stage Failed')

    return status


# Test Lasers
def test_lasers():

    laser_pass = [False, False]
    laser_color = ['green', 'red']

    for i, color in enumerate(laser_color):
        try:
            hs.laser[color].initialize()
            laser_pass[i] = True
        except:
            logger.log(21, 'Check COM Port')

    try:
        for i, color in enumerate(laser_color):
            if laser_pass[i]:
                 hs.laser[color].set_power(400)
        time.sleep(100)
        for i, color in enumerate(laser_color):
            if laser_pass[i]:
                if abs(hs.laser[color].get_power()/400-1) > 0.05:
                    warnings.warn('Laser ('+color+') unable to reach 400 mW', RuntimeWarning)
                else:
                    hs.laser[color].set_power(10)
        time.sleep(100)
        for i, color in enumerate(laser_color):
            if laser_pass[i]:
                if abs(hs.laser[color].get_power()/10-1) > 0.05:
                    warnings.warn('Laser ('+color+') unable to reach 10 mW', RuntimeWarning)
        logger.log(21, 'Lasers Nominal')
    except:
        status = False
        logger.log(21, 'Lasers Failed')

    return status

# Test Optics
def test_optics():
    try:
        hs.optics.initialize()
        status = True
    except:
        status = False
        logger.log(21, 'Lasers Failed')

    return status

# Test Temperature Control
def test_temperature_control():
    flowells = ['A', 'B']
    try:
        hs.T.initialize()
        for fc in flowcells:
            hs.T.fc_on(fc)
        for fc in flowcells:
            hs.set_fc_T(fc,50)
        for fc in flowcells:
            hs.wait_fc_T(fc,50)
        for fc in flowcells:
            hs.set_fc_T(fc,4)
        for fc in flowcells:
            hs.wait_fc_T(fc,4)
        for fc in flowcells:
            hs.fc_off(fc,4)
        logger.log(21, 'Temperature Control Nominal')
        status = True
    except:
        status = False
        logger.log(21, 'Temperature Control Failed')

# Test cameras
def test_cameras():
    try:
        hs.initializeCams(logger)
        hs.cam1.setAREA()
        hs.cam2.setAREA()
        if instrument_status['YSTAGE'] and instrument_status['FPGA']:
            image_complete = hs.take_picture(n_frames=32, image_name = 'dark')
            if image_complete == 2:
                status = True
            else:
                warnings.warn('Dark Images Failed', RuntimeWarning)
    except:
        status = False
        logger.log(21, 'Cameras Failed')


try:
    import pyseq
    hs = pyseq.HiSeq()
except:
    hs = None
    print('HiSeq Failed')

if hs is not None:
    timestamp = time.strftime('%Y%m%d%H%M')
    hs.image_path = join(os.getcwd(),timestamp+'_HiSeqCheck')
    os.mkdir(hs.image_path)
    hs.log_path = join(hs.image_path,timestamp+'_HiSeqCheck.log')
    hs.logger = setup_logger()

    instrument_tests = {'FPGA': test_led(),
                   'XSTAGE': test_x_stage(),
                   'YSTAGE': test_y_stage(),
                   'ZSTAGE': test_z_stage(),
                   'OBJSTAGE': test_objective_stage(),
                   'LASERS': test_lasers(),
                   'OPTICS': test_optics(),
                   'CAMERAS': test_cameras()}

    for instrument in instrument_tests.keys():
        instrument_status[instrument] = intruments_tests[instrument]
