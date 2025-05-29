import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert


# -------------------------------------------------------------------------------- #
#       Position your electrodes
# -------------------------------------------------------------------------------- #
class ElecPos:
    def __init__(self, n_columns, n_lines=1, h_spacing=0.85, v_spacing=0.2, start_depth=-0.3):
        """
        :param n_columns: number of column
        :param h_spacing: horizontal spacing of your electrodes
        :param n_lines: number of lines
        :param v_spacing: depth of the first electrodes
        :param start_depth: vertical spacing of your electrodes
        """
        self.n_columns = int(n_columns)
        self.h_spacing = h_spacing
        self.n_lines = int(n_lines)
        self.depth_start = start_depth
        self.v_spacing = v_spacing
        self.positions_created = False  # status indicator
        self.scheme = None  # Measurement scheme
        self.pos = None  # Positions on electrodes
        self.grid = None  # Create a grid around the electrodes
        self.jacobian = None  # jacobian matrix of the measurement protocol
        self.sensitivity_figures = None  # list of all the figures displaying sensitivity
        self.scheme_name = None  # name of the measurement protocol
        self.sensors = None
        self.ABMN = None

    def positions(self):
        self.pos = np.zeros((self.n_columns * self.n_lines, 2))
        for col in range(self.n_columns):
            for line in range(self.n_lines):
                x = round(self.h_spacing * (self.n_columns - 1) / 2 - self.h_spacing * col, 3)
                y = round(self.depth_start - line * self.v_spacing, 3)
                self.pos[col * self.n_lines + line] = [x, y]
        self.positions_created = True
        return self.pos  # Return the calculated positions

    def n_electrodes(self):
        return self.n_columns * self.n_lines

    def sensors(self):
        if not self.positions_created:
            raise RuntimeError("Positions must be defined before creating Nodes")

        sensors = []  # Initialisez la liste des capteurs
        for i in range(self.n_columns * self.n_lines):
            sensors.append([self.pos[i][0], self.pos[i][1]])  # Ajoutez les coordonnées du capteur

        self.sensors = sensors

    def createGrid(self, show=True, square_size=0.05, space=3.0):
        """
        :param show: display grid on SciView window
        :param square_size: dimensions of a square of the grid
        :param space: space around the grid (space = 6 : 6 tiles around the electrodes)
        :return:
        """
        ex = self.pos[:, 0]  # Extract x-coordinates
        ey = self.pos[:, 1]  # Extract y-coordinates

        xmin, xmax = min(ex) - space * square_size, max(ex) + space * square_size
        ymin, ymax = min(ey) - space * square_size, 0

        x = np.arange(xmin, xmax + .001, square_size)
        y = np.arange(ymin, ymax + .001, square_size)

        grid = mt.createGrid(x, y, marker=5)

        if show:
            ax, cb = pg.show(grid)
            ax.plot(pg.x(self.pos), pg.y(self.pos), "mx")
            pg.wait()

        self.grid = grid
    
    def createScheme(self, scheme_name, max_v=1,addInverse=False, jump=False):
        """
        :param scheme_name: protocol name among the following :
            ['wa','wb','pp','pd','dd','slm','hw','gr', 'ALERT', 'vertical_WA_']
        :param max_v: maximum of potential measurement made in ALERT protocol
        :return:
        """
        self.positions()  # Create electrode positions

        if scheme_name in ['wa', 'wb', 'pp', 'pd', 'dd', 'slm', 'hw', 'gr', 'uk']:
            # Create a scheme using pyGIMLi's createData
            self.scheme = ert.createData(elecs=self.pos, schemeName=scheme_name, addInverse=addInverse)
        elif scheme_name == 'ALERT':
            if max_v >= 1:
                # Create a scheme based on the ALERT protocol
                self.scheme = protocol_ALERT(self.pos, self.n_columns, self.n_lines, max_v, jump=jump)
            else:
                raise ValueError(f"max_V must be > 0: here, {max_v}")
            
        elif scheme_name == 'vertical_WA':
            self.scheme = protocol_vertical_WA(self.pos,self.n_columns, self.n_lines, addInverse=addInverse)
        elif scheme_name == 'horizontal_WA':
            self.scheme = protocol_horizontal_WA(self.pos,self.n_columns, self.n_lines, addInverse=addInverse)
        else:
            raise ValueError("Choose an existing scheme name")

        self.scheme_name = scheme_name  # Set the scheme name

    def calculateJacobian(self):
        if self.scheme is None:
            raise RuntimeError("Measurement scheme must be created before calculating the Jacobian.")
        if self.grid is None:
            raise RuntimeError("Grid must be created before calculating the Jacobian.")
        fop = ert.ERTModelling()
        fop.setData(self.scheme)
        fop.setMesh(self.grid)
        model = np.ones(self.grid.cellCount())
        fop.createJacobian(model)
        self.jacobian = fop

    def displaySensitivity(self, display_columns, save_fig=False, width_factor=4, height_factor=3):
        """
        :param save_fig: give a name to the figure
        :param display_columns: number of figures per column in the display
        :param width_factor: width of the figures (multiplied per the number of column, default = 4)
        :param height_factor: height of the figures (multiplied per the number of rows, default = 3)
        :return: a figure displaying all the sensitivity of each measurement
        """
        self.calculateJacobian()
        jacobian_matrix = self.jacobian.jacobian()
        visu.protocolSensitivity(self.grid,
                                 self.scheme,
                                 self.scheme_name,
                                 jacobian_matrix,
                                 display_columns,
                                 save_fig)
    
    def scheme_ABMN(self):
        protocol = np.empty((len(self.scheme['a']), 4), dtype=int)
        for i in range(len(protocol)):
            protocol[i] = [self.scheme['a'][i], self.scheme['b'][i], 
                           self.scheme['m'][i], self.scheme['n'][i]]
        self.ABMN = protocol

# def createGif(self, output_filename, mode='I', duration=1):


# -------------------------------------------------------------------------------- #
#       Create a measurement scheme based on the ALERT paper from Kuras O. 2009
# -------------------------------------------------------------------------------- #
# Arguments :
# -  : n_column
# - number of lines : n_lines
# - max number of measurement of potential up of the current electrodes : max_v

def protocol_ALERT(pos,
                   n_columns,
                   n_lines,
                   max_v,
                   jump=False):
    """
    :param pos: positions [x, y] of the electrodes
    :param n_columns: number of column of the electrodes array
    :param n_lines: number of lines of the electrodes array
    :param max_v: maximum of potential measurement made in ALERT protocol
    :return:
    """
    protocol = np.empty((4, 0), dtype=int)
    z = 1
    if jump:
        z = 2
    for i in range(n_columns - z):  # Loop over columns (0 to n_columns-2)
        for j in range(n_lines - 1):  # Loop over lines (0 to n_lines-2)
            for k in range(min(n_lines - j - 1, max_v)):
                # Calculate the values for the current column
                A = j + i * n_lines
                if jump:
                    B = j + 2*n_lines + i * n_lines
                else:
                    B = j + n_lines + i * n_lines
                M = A + k + 1
                N = B + k + 1
                mesure = np.array(([A, B, M, N]))[:, np.newaxis]
                protocol = np.hstack((protocol, mesure))
    scheme = ert.createData(elecs=pos, schemeName='uk')

    
    scheme['a'] = protocol[0]
    scheme['b'] = protocol[1]
    scheme['m'] = protocol[2]
    scheme['n'] = protocol[3]
    scheme["k"] = ert.createGeometricFactors(scheme)
    scheme["valid"] = np.ones(len(protocol[0]))

    return scheme

# -------------------------------------------------------------------------------- #
#       Create a measurement scheme based on the ALERT paper from Kuras O. 2009
# -------------------------------------------------------------------------------- #
# Arguments :
# -  : n_column
# - number of lines : n_lines
# - max number of measurement of potential up of the current electrodes : max_v

def protocol_vertical_WA(pos,n_columns, n_lines, addInverse=False):
    """
    :param n_columns: number of column of the electrodes array
    :param n_lines: number of lines of the electrodes array
    :return:
    """
    scheme0 = ert.createData(elecs=n_lines, schemeName='wa')
    scheme = ert.createData(elecs=pos, schemeName='uk')

    a0 = scheme0['a']
    b0 = scheme0['b']
    m0 = scheme0['m']
    n0 = scheme0['n']
    a = []
    b = []
    m = []
    n = []

    for i in range(0,n_columns):
        a = np.append(a,a0+i*n_lines)
        b = np.append(b,b0+i*n_lines)
        m = np.append(m,m0+i*n_lines)
        n = np.append(n,n0+i*n_lines)
        if addInverse:
            a = np.append(a,m0+i*n_lines)
            b = np.append(b,n0+i*n_lines)
            m = np.append(m,a0+i*n_lines)
            n = np.append(n,b0+i*n_lines)

    scheme['a'] = a
    scheme['b'] = b
    scheme['m'] = m
    scheme['n'] = n

    scheme["k"] = ert.createGeometricFactors(scheme)
    scheme["valid"] = np.ones(len(scheme["k"]))

    return scheme

def protocol_horizontal_WA(pos,n_columns, n_lines, addInverse=False):
    """
    :param n_columns: number of column of the electrodes array
    :param n_lines: number of lines of the electrodes array
    :return:
    """
    scheme0 = ert.createData(elecs=n_columns, schemeName='wa')
    scheme = ert.createData(elecs=pos, schemeName='uk')

    a0 = n_lines*scheme0['a']
    b0 = n_lines*scheme0['b']
    m0 = n_lines*scheme0['m']
    n0 = n_lines*scheme0['n']
    a = []
    b = []
    m = []
    n = []



    for i in range(0,n_lines):
        a = np.append(a,a0+i)
        b = np.append(b,b0+i)
        m = np.append(m,m0+i)
        n = np.append(n,n0+i)
        if addInverse:
            a = np.append(a,m0+i)
            b = np.append(b,n0+i)
            m = np.append(m,a0+i)
            n = np.append(n,b0+i)

    scheme['a'] = a
    scheme['b'] = b
    scheme['m'] = m
    scheme['n'] = n

    scheme["k"] = ert.createGeometricFactors(scheme)
    scheme["valid"] = np.ones(len(scheme["k"]))

    return scheme