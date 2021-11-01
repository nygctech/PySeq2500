"""PySeq Flowcell

   **Example:**

    .. code-block:: python

        #Create flowcell object
        import flowcell
        fcA = flowcell.Flowcell('A')
        # Add to ROI/section dictionary
        #section names as keys and list of bounding box as values
        fcA.sections['section1'] = [10,20,5,15]

"""



class Flowcell():
    """HiSeq 2500 System :: Flowcell

       **Attributes:**
       - position (str): Flowcell is at either position A (left slot )
         or B (right slot).
       - recipe_path (path): Path to the recipe.
       - recipe (file): File handle for the recipe.
       - first_line (int): Line number for the recipe to start from on the
         initial cycle.
       - cycle (int): The current cycle.
       - total_cycles (int): Total number of the cycles for the experiment.
       - history ([[int,],[str,],[str,]]): Timeline of flowcells events, the
         1st column is the timestamp, the 2nd column is the event, and the
         3rd column is an event specific detail.
       - sections (dict): Dictionary of section names keys and coordinate
         positions of the sections on the flowcell values.
       - stage (dict): Dictionary of section names keys and stage positioning
         and imaging details of the sections on the flowcell values.
       - thread (int): Thread id of the current event on the flowcell.
       - signal_event (str): Event that signals the other flowcell to continue
       - wait_thread (threading.Event()): Blocks other flowcell until current
         flowcell reaches signal event.
       - waits_for (str): Flowcell A waits for flowcell B and vice versa.
       - pump_speed (dict): Dictionary of pump scenario keys and pump speed
         values.
       - volume (dict): Keys are events/situations and values are volumes
         in uL to use at the event/situation.
       - filters (dict): Dictionary of filter set at each cycle, c: em, ex1, ex2.
       - IMAG_counter (None/int): Counter for multiple images per cycle.
       - events_since_IMAG (list): Record events since last IMAG step.
       - temp_timer: Timer to check temperature of flowcell.
       - temperature (float): Set temperature of flowcell in °C.
       - temp_interval (float): Interval in seconds to check flowcell temperature.
       - z_planes (int): Override number of z planes to image in recipe.
       - pre_recipe_path (path): Recipe to run before actually starting experiment
       - pre_recipe (file): File handle for the pre recipe.

    """

    def __init__(self, position):
        """Constructor for flowcells

           **Parameters:**
           - position (str): Flowcell is at either position A (left slot) or
             B (right slot).

        """

        self.recipe_path = None
        self.recipe = None
        self.first_line = None
        self.cycle = 0                                                          # Current cycle
        self.total_cycles = 0                                                   # Total number of cycles for experiment
        self.history = [[],[],[]]                                               # summary of events in flowcell history
        self.sections = {}                                                      # coordinates of flowcell of sections to image
        self.stage = {}                                                         # stage positioning info for each section
        self.thread = None                                                      # threading to do parallel actions on flowcells
        self.signal_event = None                                                # defines event that signals the next flowcell to continue
        self.wait_thread = threading.Event()                                    # blocks next flowcell until current flowcell reaches signal event
        self.waits_for = None                                                   # position of the flowcell that signals current flowcell to continue
        self.pump_speed = {'flush':700,'prime':100,'reagent':40}                # standard flowrates uL/min
        self.volume = {'main':500,'side':350,'sample':250,'flush':1000}         # standard volumes to use uL
        self.filters = {}                                                       # Dictionary of filter set at each cycle, c: em, ex1, ex2
        self.IMAG_counter = None                                                # Counter for multiple images per cycle
        self.events_since_IMAG = []                                             # List events since last IMAG step
        self.temp_timer = None                                                  # Timer to check temperature of flowcell
        self.temperature = None                                                 # Set temperature of flowcell
        self.temp_interval = None                                               # Interval in minutes to check flowcell temperature
        self.z_planes = None                                                    # Override number of z planes to image in recipe.
        self.pre_recipe_path = None                                              # Recipe to run before actually starting experiment

        while position not in ['A', 'B']:
            print('Flowcell must be at position A or B')
            position = input('Enter A or B for ' + str(position) + ' : ')

        self.position = position


    def addEvent(self, event, command):
        """Record history of events on flow cell.

           **Parameters:**
           - instrument (str): Type of event can be valv, pump, hold, wait, or
             imag.
           - command (str): Details specific to each event such as hold time,
             buffer, event to wait for, z planes to image, or pump volume.

           **Returns:**
           - int: A time stamp of the last event.

        """

        self.history[0].append(time.time())                                     # time stamp
        self.history[1].append(event)                                           # event (valv, pump, hold, wait, imag)
        self.history[2].append(command)                                         # details such hold time, buffer, event to wait for
        self.events_since_IMAG.append(event)
        if event is 'PORT':
            self.events_since_IMAG.append(command)
        if event in ['IMAG', 'STOP']:
            self.events_since_IMAG.append(event)

        return self.history[0][-1]                                              # return time stamp of last event


    def restart_recipe(self):
        """Restarts the recipe and returns the number of completed cycles."""

        # Restart recipe
        if self.recipe is not None:
            self.recipe.close()
        self.recipe = open(self.recipe_path)
        # Reset image counter (if mulitple images per cycle)
        if self.IMAG_counter is not None:
            self.IMAG_counter = 0

        msg = 'PySeq::'+self.position+'::'
        if self.cycle == self.total_cycles:
            # Increase cycle counter
            self.cycle += 1
            # Flowcell completed all cycles
            hs.message(msg+'Completed '+ str(self.total_cycles) + ' cycles')
            hs.T.fc_off(fc.position)
            self.temperature = None
            do_rinse(self)
            if self.temp_timer is not None:
                self.temp_timer.cancel()
                self.temp_timer = None
            self.thread = threading.Thread(target = time.sleep, args = (10,))
        elif self.cycle < self.total_cycles:
            # Increase cycle counter
            self.cycle += 1
            # Start new cycle
            restart_message = msg+'Starting cycle '+str(self.cycle)
            self.thread = threading.Thread(target = hs.message,
                                           args = (restart_message,))
        else:
            self.thread = threading.Thread(target = time.sleep, args = (10,))

        thread_id = self.thread.start()

        return self.cycle

    def pre_recipe(self):
        """Initializes pre recipe before starting experiment."""
        prerecipe_message = 'PySeq::'+self.position+'::'+'Starting pre recipe'
        self.recipe = open(self.prerecipe_path)
        self.thread = threading.Thread(target = hs.message,
                                       args = (prerecipe_message,))
        thread_id = self.thread.start()

        return thread_id

    def endHOLD(self):
        """Ends hold for incubations in buffer, returns False."""

        msg = 'PySeq::'+self.position+'::cycle'+str(self.cycle)+'::Hold stopped'
        hs.message(msg)

        return False
