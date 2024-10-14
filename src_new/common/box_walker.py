"""A simple box walker."""

from dm_control import mjcf
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers import legacy_base


_XML_PATH = 'box_walker.xml'


class BoxWalker(legacy_base.Walker):
    """A simple box walker."""

    def _build(
        self,
        name='walker',
        physics_timestep: float = 1e-3,
        control_timestep: float = 5e-3,
    ):
        """Build the box walker."""
        self._buffer_size = int(control_timestep // physics_timestep)
        self._mjcf_root = mjcf.from_path(_XML_PATH)
        if name:
            self._mjcf_root.model = name
        super()._build()

    def initialize_episode(self, physics, random_state):
        pass

    def apply_action(self, physics, action, random_state):
        super().apply_action(physics, action, random_state)

    def _build_observables(self):
        return BoxWalkerObservables(self, self._buffer_size)

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @composer.cached_property
    def actuators(self):
        return self._mjcf_root.find_all('actuator')

    @composer.cached_property
    def root_body(self):
        return self._mjcf_root.find('body', 'box')

    @composer.cached_property
    def end_effectors(self):
        return []

    @composer.cached_property
    def observable_joints(self):
        return []

    @composer.cached_property
    def egocentric_camera(self):
        return self._mjcf_root.find('camera', 'egocentric')

    @composer.cached_property
    def ground_contact_geoms(self):
        return self._mjcf_root.find('geom', 'torso')


class BoxWalkerObservables(legacy_base.WalkerObservables):
    """Create observables for the box walker."""

    def __init__(self, walker, buffer_size):
        self._buffer_size = buffer_size
        super().__init__(walker)

    @property
    def vestibular(self):
        """Return vestibular information."""
        return [
            self.gyro,
            self.accelerometer,
            self.velocimeter]

    @composer.observable
    def accelerometer(self):
        """Accelerometer readings."""
        return observable.MJCFFeature(
            'sensordata',
            self._entity.mjcf_model.sensor.accelerometer,
            buffer_size=self._buffer_size,
            aggregator='mean')

    @composer.observable
    def gyro(self):
        """Gyro readings."""
        return observable.MJCFFeature(
            'sensordata',
            self._entity.mjcf_model.sensor.gyro,
            buffer_size=self._buffer_size,
            aggregator='mean')

    @composer.observable
    def velocimeter(self):
        """Velocimeter readings."""
        return observable.MJCFFeature(
            'sensordata',
            self._entity.mjcf_model.sensor.velocimeter,
            buffer_size=self._buffer_size,
            aggregator='mean')

    @composer.observable
    def root_position(self):
        """Observe the box position in global coordinates."""
        return observable.MJCFFeature('xpos', self._entity.root_body)

    @composer.observable
    def root_orientation(self):
        """Observe the box orientation quaternion in global coordinates."""
        return observable.MJCFFeature('xquat', self._entity.root_body)

    @composer.observable
    def egocentric_camera(self):
        """Observable of the egocentric camera."""
        return observable.MJCFCamera(
            self._entity.egocentric_camera,
            width=64,
            height=64)
