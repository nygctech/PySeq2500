import pyseq
from pyseq.methods import userYN
from pyseq.image_analysis import get_machine_config
import logging
import time
import threading
import msvcrt


class Idle():

    def __init__(self):
        log_level = logging.DEBUG
        logger = logging.getLogger()
        logger.setLevel(10)

        c_handler = logging.StreamHandler()
        c_handler.setLevel(21)
        c_format = logging.Formatter('%(asctime)s - %(message)s', datefmt = '%Y-%m-%d %H:%M')
        c_handler.setFormatter(c_format)
        logger.addHandler(c_handler)

        self.logger = logger
        self.hs = pyseq.HiSeq(Logger = logger)
        self.pumping = False
        self.timer_thread = None
        self.camera_thread = None
        self.loop = True
        self.fc_ = ['A','B']

        self.logger.log(21,'Initializing FPGA')
        self.hs.f.initialize()
        self.start_time = None
        self.instruction_interval = 0
        self.camera_start = None
        self.port_dict = {}
        self.port_index = {'A':0, 'B':0}

        # Flash LED bar Green
        for fc in self.fc_:
            t = threading.Thread(target = self.hs.f.LED, args=(fc, 'pulse green'))
            t.start()

        # Get inlet port from user
        inlets = None
        while inlets is None:
            try:
                inlets = int(input('Pump through 2 inlet or 8 inlet port (2|8)'))
                assert inlets in [2,8]

                response = userYN(f'Confirm pumping through {inlets} port')
                if not response:
                    inlets = None
            except:
                inlets = None


        # # Ask user which port to pump from
        # self.port = None
        # while self.port is None:
        #     try:
        #         self.port = int(input('Which port to pump from (1-24)'))
        #         response = userYN(f'Confirm pumping from port {self.port}')
        #         if not response:
        #             self.port = None
        #     except:
        #         self.port = None


        # confirm flowcells ares locked
        locked = False
        while not locked:
            locked = userYN(f'Confirm A and B flowcells are both locked onto stage')


        # TODO confirm barrels per lane setting

        # Flash LED bar Green
        for fc in self.fc_:
            t = threading.Thread(target = self.hs.f.LED, args=(fc, 'green'))
            t.start()


        # Initialize Cameras
        self.logger.log(21,'Initializing Cameras')
        self.camera_thread = threading.Thread(target = self.hs.initializeCams())
        self.camera_thread.start()
        self.camera_start = time.time()

        # Initialize valves, pumps, and temperature Controller
        self.logger.log(21,'Initializing Valve and Pumps')
        initial_threads = []
        # initial_threads.append(threading.Thread(target = self.hs.T.initialize))
        for fc in self.fc_:
            initial_threads.append(threading.Thread(target = self.hs.v24[fc].initialize))
            initial_threads.append(threading.Thread(target = self.hs.v10[fc].initialize))
            initial_threads.append(threading.Thread(target = self.hs.p[fc].initialize))
            initial_threads.append(threading.Thread(target = self.hs.p[fc].update_limits, args = (8,)))
        for t in initial_threads:
            t.start()

        #While waiting get config and setup pump ports
        config, config_path = get_machine_config()
        self.config = config
        self.idle_volume = self.config.get('pump',{}).get('idle_volume', 250)

        # Wait for Valves and Pumps to initialize
        for t in initial_threads:
            t.join()


        # Set up Valves
        self.hs.move_inlet(inlets)
        for fc in self.fc_:
            ports_ = range(self.hs.v24[fc].n_ports)
            skip_ports = self.config.get('valve24',{}).get('idle_skip',[9, 20, 21, 22, 23, 24])
            self.port_dict[fc] = [p+1 for p in ports_ if  p+1 not in skip_ports]


        # # Set Chiller to 12 C
        self.logger.log(21,'Setting Chiller to 12 C')
        for i in range(3):
            self.hs.T.set_chiller_T(12, i)

        # LED -> Blue
        for fc in self.fc_:
            self.hs.f.LED(fc, 'blue')

    def pump(self, fc):

        if self.instruction_interval % 24 == 0:
            self.logger.log(21, 'Press q to quit')

        self.instruction_interval += 1
        pump_interval = int((time.time() - self.start_time) / 60)               # minutes
        self.logger.debug(f'{pump_interval} minutes since last pump')

        # Start Pump
        self.pumping = True
        port_index =  self.port_index[fc]
        port = self.port_dict[fc][port_index]
        self.hs.f.LED(fc, 'pulse blue')
        self.logger.log(21, f'Flowcell {fc} pumping from port {port}')
        self.hs.v24[fc].move(port)
        self.hs.p[fc].pump(self.idle_volume, 100)

        # Increase port index
        port_index += 1
        if port_index == len(self.port_dict[fc]):
            port_index = 0
        self.port_index[fc] = port_index

        # Stop pump
        self.hs.f.LED(fc, 'blue')
        self.pumping = False
        self.logger.log(21,f'Flowcell {fc} stopped pumping')

        if fc == 'A':
            fc = 'B'
        elif fc == 'B':
            fc = 'A'

        if self.loop:
            self.timer_thread = threading.Timer(60*60, self.pump, args = (fc,))
            self.timer_thread.start()
            self.start_time = time.time()



def main():

    idle = Idle()
    idle.start_time = time.time()
    t = threading.Thread(target = idle.pump, args = ('A',))
    t.start()

    while idle.loop:

        # Check if cameras ever initialized
        if idle.camera_start is not None and idle.camera_thread.is_alive():
            if time.time() - self.camera_start >= 3*60:
                idle.logger.log(21,'Cameras failed to initialize')
                idle.camera_start = None

        # Check for key presses
        if msvcrt.kbhit():
            key = msvcrt.getch()
            idle.logger.debug(key)
            # Quit if q is pressed
            if key.decode() == 'q':
                idle.logger.log(21,'Quitting...')
                idle.loop = False
                # Set Chiller to 4 C
                idle.logger.log(21, 'Setting Chiller back to 4 C')
                for i in range(3):
                    idle.hs.T.set_chiller_T(4, i)
                # Turn off LEDs
                idle.logger.log(21, 'turning off leds')
                for fc in idle.fc_:
                    idle.hs.f.LED(fc, 'off')
                if idle.pumping:
                    idle.logger.log(21, 'Waiting for pump to stop')
                    while idle.pumping:
                        pass
                idle.logger.log(21, 'quit complete')

    idle.logger.log(21, 'idle complete')


if __name__ == '__main__':
    main()
