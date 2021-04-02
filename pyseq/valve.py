#!/usr/bin/python
"""Illumina HiSeq 2500 System :: Valve

   Uses command set from Vici Technical Note 415

   **Example:**

    .. code-block:: python

        #Create valve object
        import pyseq
        valveA10 = pyseq.valve.Valve('COM18', name='valveA10')
        #Initialize valve
        valveA10.initialize()
        #Move valve to port #1 and confirm the valve moved to port  #1
        valveA10.move('1')
        valve10.check_valve()
        1

"""


import serial
import io
import time

# Valve object

class Valve():
    """Illumina HiSeq 2500 System :: Valve

       **Attributes:**
        - n_ports (int): Number of available ports on the valves.
        - port_dict (dict): Dictionary of port number as keys and reagent names
          as values.
        - variable_ports (list): List of ports in *port_dict* that change
          depending on the cycle
        - name (str): Name of the valve.

    """


    def __init__(self, com_port, name = None, logger = None, baudrate = 9600):
        """The constructor for the valve.

           **Parameters:**
            - com_port (str): Communication port for the valve.
            - name (str): Name of the valve.
            - logger (log, optional): The log file to write communication with the
              valve to.
            - port_dict(dict, optional): Dictionary of port number as keys and
              reagent names as values. If not specified, the port number
              is used as the reagent name.

           **Returns:**
            - valve object: A valve object to control the valve.

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
            print('ERROR::Check Valve Port')
            self.serial_port = None

        self.n_ports = 10
        self.port_dict = {}
        self.variable_ports = []
        self.side_ports = None
        self.sample_port = None
        self.prefix = ''
        self.suffix = '\r'
        self.logger = logger
        self.log_flag = False
        self.name = name



    def initialize(self):
        """Initialize the valve."""

        #Get ID of valve
        prefix = None
        while prefix == None:
            prefix = self.command('ID')                                         # Query ID number
            try:
                prefix = prefix.split('=')[1]
                prefix = prefix.replace(' ','')                                 # remove whitespace
                prefix = prefix.replace('\n','')                                # remove newline
                if prefix == 'notused':
                    prefix = ''
                self.prefix = prefix

            except:
                self.write_log('error: could not parse ID')           		    # Write error to log
                prefix = None

        #Get number of ports on valve
        n_ports = None
        while n_ports == None:
            n_ports = self.command('NP')                                        # Query Port number
            try:
                n_ports = n_ports.split('=')[1]
                self.n_ports = int(n_ports)
            except:
                self.write_log('error: could not get number of ports')          # Write error to log
                n_ports = 0

        #If port dictionary empty map 1:1
        if not self.port_dict:
            for i in range(1,self.n_ports+1):
                self.port_dict[i] = i


    def command(self, text):
        """Send a serial command to the valve and return the response.

           **Parameters:**
            - text (str): A command to send to the valve.

           **Returns:**
            - str: The response from the valve.

        """

        text = self.prefix + text + self.suffix                                 # Format the command
        self.serial_port.write(text)                                            # Write to serial port
        self.serial_port.flush()                                                #Flush serial port
        response = self.serial_port.readline()

        if self.logger is not None:                                             # Log sent command
            self.logger.info(self.name + '::txmt::'+text)
        else:
            print(text)

        blank = response
        while blank is not '':                                                  # Log received commands
            if self.logger is not None:
                self.logger.info(self.name + '::rcvd::'+blank)
            else:
                print(blank)
            blank = self.serial_port.readline()

        return  response


    def move(self, port_name):
        """Move valve to the specified port_name (str)."""

        if isinstance(port_name, int):
            if port_name in range(1,self.n_ports+1):
                position = port_name
        else:
            position = self.port_dict[port_name]

        while position != self.check_valve():
            response = self.command('GO' + str(position))
            time.sleep(1)

        return position


    def check_valve(self):
        """Return the port number position of the valve (int)."""

        position = ''
        while not position:
            position = self.command('CP')

            try:
                position = position.split('=')[1]
                position = position.replace(' ','')                             # remove whitespace
                position = position.replace('\n','')                            # remove newline
            except:
                self.write_log('error: could not parse position')               # Write error to log
                position = None

        return int(position)


    def write_log(self, text):
        """Write errors/warnings to the log"""
        if self.logger is not None:
            self.logger.info(self.name + ' ' + text)
