import numpy as np


class TimeKeeper:

    def __init__(self, days: int = 0, hours: int = 0, minutes: int = 0, **kwargs) -> None:

        self._years = 0
        self._days = days
        self._hours = hours
        self._minutes = minutes

        self.time_horizon_years: int = kwargs.get('time_horizon_years', None)
        self.max_episode_steps: int = kwargs.get('max_episode_steps', 1e3)
        self.frac_terminations : float = kwargs.get('fraction_terminated_episodes', None)
        self.timestep: int = kwargs.get('timestep', 10)

        if self.frac_terminations:

            if not self.time_horizon_years:
                raise ValueError('Time horizon must be set for termination fraction to be used.')

            episode_length = self.max_episode_steps * self.timestep # minutes
            self.f_y, r = divmod(episode_length, 365 * 24 * 60) # years
            self.f_d, r = divmod(r, 24 * 60) # days
            self.f_h, self.f_m = divmod(r, 60) # hours, minutes

    @property
    def hours(self) -> int:    
        return self._hours
    
    @property
    def minutes(self) -> int:
        return self._minutes
    
    @property
    def days(self) -> int:
        return self._days
    
    @property
    def years(self) -> int:
        return self._years
    
    @property
    def months(self) -> int:
        return self.days // 30 - 1

    def get_days(self) -> int:
            
        return self._days

    def get_hours(self) -> int:

        return self._hours
    
    def get_minutes(self) -> int:

        return self._minutes
    
    def get_years(self) -> int:

        return self._years
    
    def get_lifetime_fraction(self) -> float:

        if not self.time_horizon_years:
            return 0.0
        
        years_fraction = self._years / self.time_horizon_years
        days_fraction = (self._days / 365) / self.time_horizon_years
        hours_fraction = (self._hours / 24) / (self.time_horizon_years * 365)
        minutes_fraction = (self._minutes / 60) / (self.time_horizon_years * 365 * 24)

        return years_fraction + days_fraction + hours_fraction + minutes_fraction

    def __str__(self) -> str:

        return f'Year {self._years}, Day {self._days} - {self._hours:0>2}:{self._minutes:0>2}'
    
    def __add__(self, other: 'TimeKeeper') -> 'TimeKeeper':

        assert type(other) == type(self), f'Cannot add type {type(other)} to type {type(self)}'

        raise NotImplementedError('Addition of TimeKeeper objects is not implemented; below code does not loop around end of year, month, or day.')

        return TimeKeeper(days=self._days + other.days, hours=self._hours + other.hours, minutes=self._minutes + other.minutes, time_horizon_years=self.time_horizon_years)
    
    def update(self, hours: int = 0, minutes: int = 0) -> bool:

        self._hours += hours
        self._minutes += minutes

        if self._minutes >= 60:
            self._hours += 1
            self._minutes -= 60

        if self._hours >= 24:
            self._hours -= 24
            self._days += 1

        if self._days >= 365:
            self._days -= 365
            self._years += 1

        return self._years >= self.time_horizon_years if self.time_horizon_years else False
    
    def reset(self, random : bool = True) -> None:

        if random:

            will_terminate = np.random.rand() < self.frac_terminations if self.frac_terminations else False
            lb_y = self.time_horizon_years - 1 - self.f_y if will_terminate and self.time_horizon_years else 0
            lb_d = 364 - self.f_d if will_terminate else 0
            lb_h = 23 - self.f_h if will_terminate else 0
            lb_m = 60 - self.f_m if will_terminate else 0

            self._years = np.random.randint(lb_y, self.time_horizon_years) if self.time_horizon_years else 0
            self._days = np.random.randint(lb_d, 365)
            self._hours = np.random.randint(lb_h, 24)
            self._minutes = np.random.randint(lb_m, 60)
        else:
            self._years = 0
            self._days = 0
            self._hours = 0
            self._minutes = 0