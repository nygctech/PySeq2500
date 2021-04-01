import logging
import warnings
import time
import os
from os.path import join


def message(text):
    logger.log(21, 'PySeq::'+text)

def error(text):
    warnings.warn('ERROR::'+text, RuntimeWarning)
    if instrument_status['FPGA']:
        hs.f.LED(1, 'yellow')

def setup_logger(log_path):
    """Create a logger and return the handle."""


    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(10)

    # Create console handler
    c_handler = logging.StreamHandler()
    c_handler.setLevel(21)
    # Create file handler
    f_handler = logging.FileHandler(log_path)
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
        message('Testing LEDs')
        hs.f.initialize()
        for color in hs.f.led_dict.keys():
            for i in [1,2]:
                hs.f.LED(i, color)
            time.sleep(2)
        message('LEDs Nominal')
        status = True
    except:
        #logger.log(21, 'Check COM port')
        message('LEDs Failed')
        status = False

    return status


# Test X Stage
def test_x_stage():
    try:
        hs.y.command('OFF')
        message('Testing X Stage')
        homed = hs.x.initialize()
        if homed:
            hs.x.move(hs.x.min_x)
            hs.x.move(hs.x.max_x)
            hs.x.move(hs.x.home)
            message('X Stage Nominal')
            status = True
        else:
            error('X Stage Homing Failed')
    except:
        status = False
        #logger.log(21, 'Check COM port')
        message('X Stage Failed')

    return status

# Test Y Stage
def test_y_stage():
    try:
        message('Testing Y Stage')
        hs.y.initialize()
        start = time.time()
        timeout = 60*10
        while hs.y.check_position == 0:
            if time.time() - start > timeout:
                error('Y Stage failed to home')
                break
            else:
                time.sleep(10)
        if not hs.y.move(hs.y.min_y):
            error('Y Stage failed to move out')
        if not hs.y.move(hs.y.max_y):
            error('Y Stage failed to move in')
        if not hs.y.move(hs.y.home):
            error('Y stage failed to move home')
        if instrument_status['FPGA'] and hs.x.check_position(hs.x.home):
            attempts = 0
            if abs(hs.f.read_position() - hs.y.read_position()) > 10:
                hs.reset_stage()
                attempts += 1
                if attempts >= 10:
                    error('Unable to sync FPGA & Y Stage')

        status = True
        message('Y Stage Nominal')

    except:
        status = False
        #logger.log(21, 'Check COM port')
        message('Y Stage Failed')

    return status

# Test Z Stage
def test_z_stage():
    z_pass = [False, False, False]
    try:
        message('Testing Z Stage')
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
            message('Z Stage Nominal')
        else:
            for i, z in enumerate(z_pass):
                if not z:
                    error('Z Tilt Motor '+str(i)+' Failed.')
    except:
        status = False
        message('Z Stage Failed')

    return status


# Test Objective Stage
def test_objective_stage():
    try:
        message('Testing Objective Stage')
        hs.obj.initialize()
        hs.obj.move(hs.obj.focus_start)
        hs.obj.move(hs.obj.focus_stop)
        hs.obj.set_velocity(hs.obj.min_v)
        hs.obj.move(hs.obj.focus_start)
        hs.obj.move(hs.obj.focus_stop)
        hs.obj.set_velocity(hs.obj.max_v)
        message('Objective Stage Nominal')
        status = True
    except:
        status = False
        message('Objective Stage Failed')

    return status


# Test Lasers
def test_lasers():
    message('Testing Lasers')

    laser_pass = [False, False]
    laser_color = ['green', 'red']

    for i, color in enumerate(laser_color):
        try:
            hs.lasers[color].initialize()
            laser_pass[i] = True
        except:
            error('Check COM port assignment for '+color+' laser')

    try:
        for i, color in enumerate(laser_color):
            if laser_pass[i]:
                 hs.lasers[color].set_power(400)
        timeout = 10*60
        start = time.time()
        keep_going = [True, True]
        while any(keep_going):
            for i, color in enumerate(laser_color):
                if laser_pass[i]:
                    if abs(hs.lasers[color].get_power()/400-1) > 0.05:
                        if time.time()-start > timeout:
                            laser_pass[i] = False
                            keep_going[i] = False
                            error('Laser ('+color+') unable to reach 400 mW')
                    else:
                        keep_going[i] = False
                        hs.lasers[color].set_power(10)
                else:
                    keep_going[i] = True

        start = time.time()
        keep_going = [True, True]
        while any(keep_going):
            for i, color in enumerate(laser_color):
                if laser_pass[i]:
                    if abs(hs.lasers[color].get_power()/10-1) > 0.05:
                        laser_pass[i] = False
                        keep_going[i] = False
                        error('Laser ('+color+') unable to reach 10 mW')
                    else:
                        keep_going[i] = False
                else:
                    keep_going[i] = False

        if all(laser_pass):
            status = True
            message('Lasers Nominal')
    except:
        status = False
        message('Lasers Failed')

    return status

# Test Optics
def test_optics():
    message('Testing Optics')
    try:
        hs.optics.initialize()
        status = True
        message('Optics Nominal')
    except:
        status = False
        message('Optics Failed')

    return status

# Test Temperature Control
def test_temperature_control():
    message('Testing Stage Temperature Control')
    flowcells = ['A', 'B']
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
        message('Temperature Control Nominal')
        status = True
    except:
        status = False
        message('Temperature Control Failed')

# Test cameras
def test_cameras():
    message('Testing Cameras')
    try:
        hs.initializeCams(logger)
        hs.cam1.setAREA()
        hs.cam2.setAREA()
        if instrument_status['YSTAGE'] and instrument_status['FPGA']:
            image_complete = hs.take_picture(n_frames=32, image_name = 'dark')
            if image_complete:
                status = True
                message('Cameras Nominal')
            else:
                error('Dark Images Failed')
        else:
            error('Unable to image without Y Stage and FPGA')
    except:
        status = False
        message('Cameras Failed')




timestamp = time.strftime('%Y%m%d%H%M')
image_path = join(os.getcwd(),timestamp+'_HiSeqCheck')
os.mkdir(image_path)
log_path = join(image_path,timestamp+'_HiSeqCheck.log')
logger = setup_logger(log_path)

try:
    import pyseq
    hs = pyseq.HiSeq(logger)
    # Exception for ValueError of port, must be string or None, not int)
    # Exception for SerialException, could not open port
    hs.image_path = image_path
except:
    hs = None
    message('HiSeq Failed')

if hs is not None:
    instrument_tests = {'FPGA': test_led,
                   'XSTAGE': test_x_stage,
                   'YSTAGE': test_y_stage,
                   'ZSTAGE': test_z_stage,
                   'OBJSTAGE': test_objective_stage,
                   'LASERS': test_lasers,
                   'OPTICS': test_optics,
                   'TEMPERATURE': test_temperature_control,
                   'CAMERAS': test_cameras}

    instrument_status = {'FPGA':False}

    for instrument in instrument_tests.keys():
        if instrument_status['FPGA']:
            hs.f.LED('A', 'green')
            hs.f.LED('B', 'off')

        instrument_status[instrument] = instrument_tests[instrument]()

        if instrument_status[instrument] and instrument_status['FPGA']:
            hs.f.LED('B', 'green')
            time.sleep(2)

    hs.f.LED('A', 'pulse green')
    hs.f.LED('B', 'pulse green')

    table = []
    for instrument in instrument_status.keys():
        if instrument_status[instrument]:
            table.append([instrument, 'PASSED'])
        else:
            table.append([instrument, 'FAILED'])
    try:
        import tabulate
        print(tabulate.tabulate(table, tablefmt = 'presto'))
    except:
        print(table)
