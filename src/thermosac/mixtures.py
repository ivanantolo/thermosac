import numpy as np
from typing import Tuple, Iterator, Optional
from numbers import Number
from copy import copy
from .constants import kb, R


class Component:
    r'''
    Object class for storing pure component information.

    Parameters
    ----------
    name : str
        Name of the component
    Tc : float
        Critical temperature [K]
    Pc : float
        Critical pressure [bar]
    Zc : float
        Critical compressibility factor
    Vc : float
        Critical molar volume [:math:`\mathrm{cm^3/mol}`]
    w  : float
        Acentric factor
    c : float
        Volume translation parameter used in cubic EoS [:math:`\mathrm{cm^3/mol}`]
    cii : List[float]
        Polynomial coefficients for influence parameter used in SGT model
    ksv : List[float]
        Parameter for alpha for PRSV EoS
    Ant : List[float]
        Antoine correlation parameters
    GC : dict
        Group contribution information used in Modified-UNIFAC
        activity coefficient model. Group definitions can be found `here
        <http://www.ddbst.com/PublishedParametersUNIFACDO.html#ListOfMainGroups>`_.
    Mw : float
        molar weight of the fluid [g/mol]
    ri : float
        Component molecular volume for UNIQUAC model
    qi : float
        Component molecular surface for UNIQUAC model

    '''

    def __init__(self,
                 name: Optional[str] = None,
                 Tc=0, Pc=0, Zc=0, Vc=0, w=0, c=0,
                 cii=0, ksv=[0, 0], Ant=[0, 0, 0],  GC=None,
                 # Mw: Optional[Number] = 1.,  # Positive number expected
                  Mw: Optional[Number] = None,  # Positive number expected
                 ri=0., qi=0., ms=1, sigma=0, eps=0, lambda_r=12.,
                 lambda_a=6., eAB=0., rcAB=1., rdAB=0.4, sites=[0, 0, 0],
                 T_fus: Optional[Number] = None,  # Positive number expected
                 H_fus: Number = 0,
                 Cp_fus_a_fit: Number = 0,
                 Cp_fus_bT_fit: Number = 0,
                 v_298: Optional[Number] = None,
                 v_hc: Optional[Number] = None,
                 sigma_file='None',
                 path_to_sigma_profile='.',
                 mu0=0,
                 **kwargs):

        self.name = name
        self.Tc = Tc  # Critical Temperature in K
        self.Pc = Pc  # Critical Pressure in bar
        self.Zc = Zc  # Critical compressibility factor
        self.Vc = Vc  # Critical volume in cm3/mol
        if Vc == 0 and Zc != 0:
            self.Vc = R*Zc*Tc/Pc
        elif Vc != 0 and Zc == 0:
            self.Zc = Pc*Vc/(R*Tc)
        self.w = w  # Acentric Factor
        self.Ant = Ant  # Antoine coefficeint, base e = 2.71
        self.cii = cii  # Influence factor SGT, list or array
        self.ksv = ksv
        self.c = c  # volume translation for cubic EoS
        self.GC = GC  # Dict, Group contribution info
        # self.nc = 1
        self.Mw = Mw  # molar weight in g/mol
        self.ri = ri  # Component molecular volume for UNIQUAC model
        self.qi = qi  # Component molecular surface for UNIQUAC model
        # for SLE calculations
        self.T_fus = T_fus
        self.H_fus = H_fus
        self.Cp_fus_a_fit = Cp_fus_a_fit
        self.Cp_fus_bT_fit = Cp_fus_bT_fit
        self.v_298 = v_298
        self.v_hc = v_hc
        # for COSMOSAC
        self.sigma_file = sigma_file  # name if sigma_file=='None' else sigma_file
        self.path_to_sigma_profile = kwargs.get(
            'p2sigma', path_to_sigma_profile)
        self.p2sigma = self.path_to_sigma_profile  # shorthand version
        # for Chemical potential fitting
        self.mu0 = mu0

    def __add__(self, component2):
        '''
        Methods to add two components and create a mixture with them.
        '''
        return Mixture(self, component2)

    def __repr__(self):
        return f"<Component('{self.name}')>"

    def __str__(self):
        return f"{self.name}"

    def __eq__(self, other):
        if isinstance(other, Component):
            # Here you can define what makes two instances equal.
            # For example, let's compare the name, Tc, Pc, and Mw.
            return (self.name == other.name and
                    self.Tc == other.Tc and
                    self.Pc == other.Pc and
                    self.Mw == other.Mw)
        return NotImplemented


class Mixture:
    components: Tuple[Component]

    def __init__(self, *components):
        self.components = components
        self.aggregate_properties()
        self.update_properties = self.aggregate_properties

    def aggregate_properties(self):
        # This will store lists of properties, keyed by property name
        aggregated = {}
        convert_to_plural = [
            'name',
            'sigma_file',
            'path_to_sigma_profile',
        ]

        # Iterate over all components
        for comp in self.components:
            # Iterate over all attributes of the current component
            for attr_name, attr_value in comp.__dict__.items():
                if attr_name not in aggregated:
                    aggregated[attr_name] = []
                aggregated[attr_name].append(attr_value)

        # Now, assign these lists as attributes on the Mixture instance itself
        for attr_name, attr_list in aggregated.items():
            if attr_name in convert_to_plural:
                setattr(self, attr_name + 's', list(attr_list))
            else:
                # Make arrays be size (n,1) instead of (n,)
                # (1) For the convert(x, to='weight') function
                # (2) For the gibbs_fusion(T) function of SLE
                # setattr(self, attr_name, np.array(attr_list)[:, np.newaxis])
                setattr(self, attr_name, np.array(attr_list))

        # Assign attributes manually
        self.nc = len(self.components)
        self.p2sigma = self.path_to_sigma_profiles

    def add_component(self, component):
        self.components += (component,)
        self.aggregate_properties()

    def get_component_by_name(self, name):
        """Retrieve a component by its name.

        Parameters:
        name (str): The name of the component to retrieve.

        Returns:
        Component: The component object with the given name.
        """
        index = self.names.index(name)
        return self[index]


    def NRTL(self, alpha, T12, T21):
        self.alpha, self.T12, self.T21 = alpha, T12, T21
        # self.alpha      = np.array([[0., alpha], [alpha, 0.]])
        # self.tau        = np.array([[0., T12  ], [T21,   0.]])
        # self.G          = np.exp(-alpha*tau)
        # self.actmodelp  = (self.alpha, self.tau, self.G)
        self.actmodelp = (self.alpha, self.T12, self.T21)

    def COSMOSAC(self, alpha, T12, T21):
        self.alpha, self.T12, self.T21 = alpha, T12, T21
        self.actmodelp = (self.alpha, self.T12, self.T21)



    def __add__(self, new_component):
        if isinstance(new_component, Component):
            self.add_component(new_component)
        else:
            raise Exception(
                'You can only add "Components" objects to an existing mixture')
        return self

    def copy(self):
        """
        Returns a copy of the mixture object
        """
        return copy(self)

    def __repr__(self):
        component_names = ', '.join(
            [f"'{comp.name}'" for comp in self.components])
        return f"<Mixture({component_names})>"

    def __str__(self):
        component_names = ' + '.join(
            [f"{comp.name}" for comp in self.components])
        return f"{component_names}"

    def __iter__(self) -> Iterator[Component]:
        """
        Makes the mixture object iterable (i.e., one can loop over it)
        """
        return iter(self.components)
        # return iter([Mixture(comp) for comp in self.components])

    def __getitem__(self, index):
        return self.components[index]
