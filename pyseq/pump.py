#!/usr/bin/python
"""Illumina HiSeq 2500 System :: Pump

   Uses command set from Kloehn VersaPump3

   **Example:**

    .. code-block:: python

        #Create pump object
        import pyseq
        pumpA = pyseq.pump.Pump('COM10','pumpA')
        #Initialize pump
        pumpA.initialize()
        #Pump 2000 uL at 4000 uL/min
        pumpA.pump(2000,4000)

    Max volume and max/min flowrate (Q) pumped to each lane on a flowcell depend
    on the HiSeq plumbing configuration ie how many pump barrels are tied to a
    flowcell lane. The minimum volume regardless of the plumbing configuration
    is 1 uL.

    ============   ===============   ==============  ==============
    barrels/lane   Max Volume (uL)   Max Q (uL/min)  Min Q (uL/min)
    ============   ===============   ==============  ==============
    8              2000              20000           100
    4              1000              10000           50
    2              500               5000            25
    1              250               2500            13
    ============   ===============   ==============  ==============

"""


import serial
import io
import time


class Pump():
    """HiSeq 2500 System :: Pump

       **Attributes:**
        - n_barrels (int): The number of barrels used per lane. The max is 8.
        - max_volume (float): The maximum volume pumped in one stroke in uL.
        - min_volume (float): The minimum volume that can be pumped in uL.
        - max_flow (int): The maximum flowrate of the pump in uL/min.
        - min_flow (int): The minimum flowrate of the pump in uL/min.
        - dispense_speed (int): The speed to dipense liquid to waste in steps
          per second.
        - delay (int): Seconds to wait before switching valve.
        - prefix (str): The prefix for commands to the pump. It depends on the
          pump address.
        - name (str): The name of the pump.

    """


    def __init__(self, com_port, name=None, logger=None, baudrate = 9600):
        """The constructor for the pump.

           **Parameters:**
            - com_port (str): Communication port for the pump.
            - name (str): Name of the pump.
            - logger (log, optional): The log file to write communication with the
              pump to.
            - n_barrels (int): The number of barrels used per lane (max = 8,
              default = 1)

           **Returns:**
            - pump object: A pump object to control the pump.

        """



        if isinstance(com_port, int):
            com_port = 'COM'+str(com_port)

        try:
            # Open Serial Port
            s  = serial.Serial(com_port, baudrate, timeout = 1)
            # Text wrapper around serial port
            self.serial_port = io.TextIOWrapper(io.BufferedRWPair(s,s,),
                                                encoding = 'ascii',
                                                errors = 'ignore')
        except:
            print('ERROR::Check Pump Stage Port')
            self.serial_port = None

        self.n_barrels = 1
        self.barrel_volume = 250.0 # uL
        self.steps = 48000.0
        self.max_volume = self.n_barrels*self.barrel_volume #uL
        self.min_volume = self.max_volume/self.steps #uL
        self.min_flow = int(self.min_volume*40*60) # uL per min (upm)
        self.max_flow = int(self.min_volume*8000*60) # uL per min (upm)
        self.dispense_speed = 7000 # speed to dispense (sps)
        self.delay = 10 # wait 10 s before switching valve
        self.prefix = '/1'
        self.suffix = '\r'
        self.logger = logger
        self.name = name


    def initialize(self):
        """Initialize the pump."""

        response = self.command('W4R')                                          # Initialize Stage

    def update_limits(self, n_barrels):
        """Change barrels/flowcell lane and update volume and flowrate limits."""

        self.n_barrels = n_barrels
        self.max_volume = self.n_barrels*self.barrel_volume #uL
        self.min_volume = self.max_volume/self.steps #uL
        self.min_flow = int(self.min_volume*40*60) # uL per min (upm)
        self.max_flow = int(self.min_volume*8000*60) # uL per min (upm)

    def command(self, text):
        """Send a serial command to the pump and return the response.

           **Parameters:**
            - text (str): A command to send to the pump.

           **Returns:**
            - str: The response from the pump.

        """

        text = self.prefix + text + self.suffix                                 # Format text
        self.serial_port.write(text)                                            # Write to serial port
        self.serial_port.flush()                                                # Flush serial port
        response = self.serial_port.readline()                                  # Get the response
        if self.logger is not None:                                             # Write command and response to log
            self.logger.info(self.name+'::txmt::'+text)
            self.logger.info(self.name+'::rcvd::'+response)

        return  response


    def pump(self, volume, flow = 0):
        """Pump desired volume at desired flowrate then send liquid to waste.

           **Parameters:**
            - volume (float): The volume to be pumped in uL.
            - flow (float): The flowrate to pump at in uL/min.

        """

        if flow == 0:
            flow = self.flow                                              # Default to min speed

        position = self.vol_to_pos(volume)                                      # Convert volume (uL) to position (steps)
        sps = self.uLperMin_to_sps(flow)                                       # Convert flowrate(uLperMin) to steps per second

        self.check_pump()                                                       # Make sure pump is ready

        #Aspirate
        while position != self.check_position():
            self.command('IV' + str(sps) + 'A' + str(position) + 'R')           # Pull syringe down to position
            self.check_pump()

        time.sleep(self.delay)
        self.command('OR')                                                      # Switch valve to waste

        #Dispense
        position = 0
        while position != self.check_position():
            self.command('OV' + str(self.dispense_speed) + 'A0R')               # Dispense, Push syringe to top at dispense speed
            self.check_pump()
        self.command('IR')                                                      # Switch valve to input



    def reverse_pump(self, volume, in_flow=0, out_flow=0):
        """Pump from outlet and then send liquid to inlet.

           **Parameters:**
            - volume (float): The volume to be pumped in uL.
            - in_flow (float): The flowrate to aspirate from waste in uL/min.
            - out_flow (float): The flowrate to dispense to inlet in uL/min.

        """

        if in_flow == 0:
            in_flow = 1000                                                      # Default to 1000 uL/min
        if out_flow == 0:
            out_flow = 100                                                      # Default to 100 uL/min

        position = self.vol_to_pos(volume)                                      # Convert volume (uL) to position (steps)
        in_sps = self.uLperMin_to_sps(in_flow)                                  # Convert flowrate(uLperMin) to steps per second
        out_sps = self.uLperMin_to_sps(out_flow)

        self.check_pump()                                                       # Make sure pump is ready

        #Aspirate
        while position != self.check_position():
            self.command('OV' + str(in_sps) + 'A' + str(position) + 'R')        # Pull syringe down to position
            self.check_pump()

        self.command('IR')                                                      # Switch valve to inlet

        #Dispense
        position = 0
        while position != self.check_position():
            self.command('IV' + str(out_sps) + 'A0R')                           # Dispense, Push syringe to top at dispense speed
            self.check_pump()

        self.command('IR')


    def check_pump(self):
        """Wait until pump is ready and then return True.

           **Returns:**
            - bool: True when the pump is ready. False, if the pump has an error.

        """

        busy = '@'
        ready = '`'
        status_code = ''

        while status_code != ready :

            while not status_code:
                status_code = self.command('')                                  # Ping pump for status

                if status_code.find(busy) > -1:
                    status_code = ''
                    time.sleep(2)
                elif status_code.find(ready) > -1:
                    status_code = ready
                    return True
                else:
                    self.write_log('WARNING:Pump error')
                    return False


    def check_position(self):
        """Return the pump position.

           **Returns:**
            - int: The step position of the pump (0-48000).

        """

        pump_position = self.command('?')

        while not isinstance(pump_position, int):
            pump_position = self.command('?')
            try:
                pump_position = pump_position.split('`')[1]
                pump_position = int(pump_position.split('\x03')[0])
            except:
                self.write_log('ERROR::Could not parse position')
                pump_position = None

        return pump_position


    def vol_to_pos(self, volume):
        """Convert volume from uL (float) to pump position (int, 0-48000).

           If the volume is too big or too small, returns the max or min volume.

        """

        if volume > self.max_volume:
            self.write_log('Volume is too large, only pumping ' +
                       str(self.max_volume))
            volume = self.max_volume
        elif volume < self.min_volume:
            self.write_log('Volume is too small, pumping ' +
                       str(self.min_volume))
            volume = self.min_volume

        position = round(volume / self.max_volume * self.steps)
        return int(position)


    def uLperMin_to_sps(self, flow):
        """Convert flowrate from uL per min. (float) to steps per second (int).

        """

        if flow < self.min_flow:
            flow = self.min_flow
            self.write_log('Flowrate is too slow, increased to ' + str(flow) +
                           ' uL/min')
        elif flow > self.max_flow:
            flow = self.max_flow
            self.write_log('Flowrate is too fast, decreased to ' + str(flow) +
                           ' uL/min')

        sps = round(flow / self.min_volume / 60)

        return int(sps)


    def write_log(self, text):
        """Write messages to the log."""

        if self.logger is None:
            print(self.name + ' ' + text)
        else:
            self.logger.info(self.name + ' ' + text)
