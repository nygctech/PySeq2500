import logging
import warnings
import time
import os
import sys
from os.path import join



def message(text):
    logger.log(21, 'PySeq::'+text)

def error(text):
    message(text)
    if instrument_status['FPGA']:
        hs.f.LED(1, 'yellow')
    raise RuntimeError

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
    hs.y.command('Z')
    hs.y.command('OFF')
    try:

        message('Testing X Stage')
        homed = hs.x.initialize()
        if homed:
            hs.x.move(hs.x.min_x)
            hs.x.move(hs.x.max_x)
            hs.x.move(hs.x.home)
            message('X Stage Nominal')
            status = True
        else:
            text = 'X Stage Homing Failed'
            error(text)
            #warnings.warn('ERROR::'+text, RuntimeWarning)

    except:
        #error()
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
        while hs.y.check_position() == 0:
            if time.time() - start > timeout:
                text = 'Y Stage failed to home'
                error(text)
                #warnings.warn('ERROR::'+text, RuntimeWarning)
                break
            else:
                time.sleep(10)
        if not hs.y.move(hs.y.min_y):
            text = 'Y Stage failed to move out'
            error(text)
            #warnings.warn('ERROR::'+text, RuntimeWarning)
        if not hs.y.move(hs.y.max_y):
            text = 'Y Stage failed to move in'
            error(text)
            #warnings.warn('ERROR::'+text, RuntimeWarning)
        if not hs.y.move(hs.y.home):
            text = 'Y stage failed to move home'
            error(text)
            #warnings.warn('ERROR::'+text, RuntimeWarning)
        if instrument_status['FPGA'] and hs.x.check_position(hs.x.home):
            attempts = 0
            if abs(hs.f.read_position() - hs.y.read_position()) > 10:
                hs.reset_stage()
                attempts += 1
                if attempts >= 10:
                    text = 'Unable to sync FPGA & Y Stage'
                    error(text)
                    #warnings.warn('ERROR::'+text, RuntimeWarning)

        status = True
        message('Y Stage Nominal')

    except:
        #error()
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
                    text = 'Z Tilt Motor '+str(i)+' Failed.'
                    warnings.warn('ERROR::'+text, RuntimeWarning)

    except:
        error()
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
            text = 'Laser ('+color+') unable to initialize'
            warnings.warn('ERROR::'+text, RuntimeWarning)

        if laser_pass[i]:
            try:
                hs.lasers[color].set_power(400)
                timeout = 10*60
                start = time.time()
                keep_waiting = True
                while keep_waiting:
                    if abs(hs.lasers[color].get_power()/400-1) > 0.05:
                        if time.time()-start > timeout:
                            keep_waiting = False
                            text = 'Laser ('+color+') unable to reach 400 mW'
                            warnings.warn('ERROR::'+text, RuntimeWarning)
                    else:
                        keep_waiting = False
                        hs.lasers[color].set_power(10)

            except:
                error()
                laser_pass[i] = False
                message(color + ' Laser Failed')

    if all(laser_pass):
        status = True
        message('Lasers Nominal')
    else:
        status = False

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
            hs.T.set_fc_T(fc,50)
        for fc in flowcells:
            hs.T.wait_fc_T(fc,50)
        for fc in flowcells:
            hs.T.set_fc_T(fc,20)
        for fc in flowcells:
            hs.T.wait_fc_T(fc,20)
        for fc in flowcells:
            hs.T.fc_off(fc)
        message('Temperature Control Nominal')
        status = True
    except:
        status = False
        message('Temperature Control Failed')

    return status

# Test valves
def test_valves():
    valve_pass = [False, False, False, False]

    message('Testing Valves')

    for i, AorB in enumerate(['A','B']):
        try:
            hs.v24[AorB].initialize()
            hs.v24[AorB].move(1)
            valve_pass[i] = True
        except:
            error()
            message('24 Port ' + AorB + ' Valve Failed')

    for i, AorB in enumerate(['A','B']):
        try:
            hs.v10[AorB].initialize()
            valve_pass[i+2] = True
        except:
            error()
            message('10 Port ' + AorB + ' Valve Failed')
    try:
        hs.move_inlet(8)
    except:
        valve_pass[2] = False
        valve_pass[3] = False
        error()
        message('Error moving 10 Port Valves')

    if all(valve_pass):
        status = True
        message('Valves Nominal')
    else:
        status = False

    return status

# Test Pumps
def test_pumps():

    pump_pass = [False, False]
    message('Testing Pumps')
    for i, AorB in enumerate(['A','B']):
        try:
            hs.p[AorB].initialize()
            if hs.p[AorB].check_pump():
                pump_pass[i] = True
            else:
                text = 'Pump ' + AorB + ' Error'
                warnings.warn('ERROR::'+text, RuntimeWarning)
        except:
            error()
            pump_pass[i] = False
            message('Pump ' + AorB + ' Failed')

    if all(pump_pass):
        status = True
        message('Pumps Nominal')
    else:
        status = False

    return status


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
                text = 'Dark Images Failed'
                warnings.warn('ERROR::'+text, RuntimeWarning)
        else:
            text = 'Unable to image without Y Stage and FPGA'
            error(text)
            #raise RuntimeError(text)
            #warnings.warn('ERROR::'+text, RuntimeWarning)
    except:
        #error()
        status = False
        message('Cameras Failed')

    return status






try:
    timestamp = time.strftime('%Y%m%d%H%M')
    image_path = join(os.getcwd(),timestamp+'_HiSeqCheck')
    os.mkdir(image_path)
    log_path = join(image_path,timestamp+'_HiSeqCheck.log')
    logger = setup_logger(log_path)


    # Creat HiSeq Object
    import pyseq
    com_ports = pyseq.get_com_ports()
    hs = pyseq.HiSeq(logger, com_ports)
    # Exception for ValueError of port, must be string or None, not int)
    # Exception for SerialException, could not open port
    hs.image_path = image_path

except ImportError:
    message('PySeq Failed')
    sys.exit()
except OSError:
    message('Failed to make directories')
    sys.exit()
# except:
#     message('HiSeq Failed')
#     sys.exit()


instrument_tests = {'FPGA': test_led,
               'XSTAGE': test_x_stage,
               'YSTAGE': test_y_stage,
               'ZSTAGE': test_z_stage,
               'OBJSTAGE': test_objective_stage,
               'LASERS': test_lasers,
               'OPTICS': test_optics,
               'TEMPERATURE': test_temperature_control,
               'VALVES': test_valves,
               'PUMPS': test_pumps,
               'CAMERAS': test_cameras}

instrument_status = {'FPGA':False}

for instrument in instrument_tests.keys():
    if instrument_status['FPGA']:
        hs.f.LED('A', 'green')
        hs.f.LED('B', 'pulse blue')

    instrument_status[instrument] = instrument_tests[instrument]()

    if instrument_status[instrument] and instrument_status['FPGA']:
        hs.f.LED('B', 'green')
        time.sleep(2)

if instrument_status['FPGA']:
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
    print('\n')
    print(tabulate.tabulate(table, headers = ['INSTRUMENT', 'STATUS'],
                                  tablefmt = 'presto'))
except:
    print(table)
