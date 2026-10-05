import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional


class InvalidStateTransitionError(Exception):
    pass


class DriverStatus(Enum):
    AVAILABLE = auto()
    ON_TRIP = auto()
    OFFLINE = auto()


class TripStatus(Enum):
    REQUESTED = auto()
    ACCEPTED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    CANCELLED = auto()


@dataclass(frozen=True)
class Location:
    x: float
    y: float

    def distance_to(self, other: "Location") -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


class Vehicle(ABC):
    def __init__(self, reg_no: str, capacity: int, rate_multiplier: float):
        self.reg_no = reg_no
        self.capacity = capacity
        self.rate_multiplier = rate_multiplier
        self.vehicle_type = self.__class__.__name__


class BikeVehicle(Vehicle):
    def __init__(self, reg_no: str):
        super().__init__(reg_no=reg_no, capacity=1, rate_multiplier=0.8)


class SedanVehicle(Vehicle):
    def __init__(self, reg_no: str):
        super().__init__(reg_no=reg_no, capacity=4, rate_multiplier=1.0)


class SuvVehicle(Vehicle):
    def __init__(self, reg_no: str):
        super().__init__(reg_no=reg_no, capacity=6, rate_multiplier=1.5)


class Rider:
    def __init__(self, rider_id: str, name: str, contact: str):
        self.id = rider_id
        self.name = name
        self.contact = contact


class Driver:
    def __init__(self, driver_id: str, name: str, vehicle: Vehicle, location: Location):
        self.id = driver_id
        self.name = name
        self.vehicle = vehicle
        self.location = location
        self.status = DriverStatus.AVAILABLE
        self.rating: float = 5.0
        self._rating_count: int = 1
        self._rating_sum: float = 5.0

    def update_location(self, new_location: Location) -> None:
        self.location = new_location

    def add_rating(self, score: float) -> None:
        self._rating_count += 1
        self._rating_sum += score
        self.rating = round(self._rating_sum / self._rating_count, 2)


class Trip:
    def __init__(
        self,
        trip_id: str,
        rider: Rider,
        driver: Driver,
        pickup: Location,
        drop: Location,
        fare: float,
    ):
        self.trip_id = trip_id
        self.rider = rider
        self.driver = driver
        self.pickup = pickup
        self.drop = drop
        self.fare = fare
        self.status = TripStatus.REQUESTED

    def start_trip(self) -> None:
        if self.status != TripStatus.ACCEPTED:
            raise InvalidStateTransitionError(
                f"Cannot start trip from state {self.status.name}. Must be ACCEPTED."
            )
        self.status = TripStatus.IN_PROGRESS

    def complete_trip(self) -> None:
        if self.status != TripStatus.IN_PROGRESS:
            raise InvalidStateTransitionError(
                f"Cannot complete trip from state {self.status.name}. Must be IN_PROGRESS."
            )
        self.status = TripStatus.COMPLETED
        self.driver.status = DriverStatus.AVAILABLE
        self.driver.location = self.drop

    def cancel_trip(self) -> None:
        if self.status not in (TripStatus.REQUESTED, TripStatus.ACCEPTED):
            raise InvalidStateTransitionError(
                f"Cannot cancel trip in state {self.status.name}."
            )
        self.status = TripStatus.CANCELLED
        self.driver.status = DriverStatus.AVAILABLE


class DriverMatchingStrategy(ABC):
    @abstractmethod
    def match_driver(
        self, pickup: Location, drivers: List[Driver], vehicle_type: str
    ) -> Optional[Driver]:
        pass


class NearestDriverStrategy(DriverMatchingStrategy):
    def match_driver(
        self, pickup: Location, drivers: List[Driver], vehicle_type: str
    ) -> Optional[Driver]:
        eligible = [
            d for d in drivers
            if d.status == DriverStatus.AVAILABLE and d.vehicle.vehicle_type == vehicle_type
        ]
        if not eligible:
            return None
        return min(eligible, key=lambda d: d.location.distance_to(pickup))


class HighestRatedDriverStrategy(DriverMatchingStrategy):
    def __init__(self, max_radius_km: float = 5.0):
        self.max_radius_km = max_radius_km

    def match_driver(
        self, pickup: Location, drivers: List[Driver], vehicle_type: str
    ) -> Optional[Driver]:
        eligible = [
            d for d in drivers
            if d.status == DriverStatus.AVAILABLE
            and d.vehicle.vehicle_type == vehicle_type
            and d.location.distance_to(pickup) <= self.max_radius_km
        ]
        if not eligible:
            return None
        return max(eligible, key=lambda d: d.rating)


class PricingStrategy(ABC):
    @abstractmethod
    def calculate_fare(self, distance: float, vehicle: Vehicle) -> float:
        pass


class BasePricingStrategy(PricingStrategy):
    def __init__(self, base_fare: float = 5.0, rate_per_km: float = 2.0):
        self.base_fare = base_fare
        self.rate_per_km = rate_per_km

    def calculate_fare(self, distance: float, vehicle: Vehicle) -> float:
        raw_fare = (self.base_fare + (distance * self.rate_per_km)
                    ) * vehicle.rate_multiplier
        return round(raw_fare, 2)


class SurgePricingStrategy(PricingStrategy):
    def __init__(
        self, surge_multiplier: float = 1.5, base_fare: float = 5.0, rate_per_km: float = 2.0
    ):
        self.surge_multiplier = surge_multiplier
        self.base_fare = base_fare
        self.rate_per_km = rate_per_km

    def calculate_fare(self, distance: float, vehicle: Vehicle) -> float:
        base_calc = (self.base_fare + (distance * self.rate_per_km)
                     ) * vehicle.rate_multiplier
        return round(base_calc * self.surge_multiplier, 2)


class RideServiceManager:
    def __init__(
        self,
        matching_strategy: DriverMatchingStrategy,
        pricing_strategy: PricingStrategy,
    ):
        self.matching_strategy = matching_strategy
        self.pricing_strategy = pricing_strategy
        self.riders: dict[str, Rider] = {}
        self.drivers: dict[str, Driver] = {}
        self.trips: dict[str, Trip] = {}
        self._trip_counter = 0

    def register_rider(self, rider_id: str, name: str, contact: str) -> Rider:
        rider = Rider(rider_id, name, contact)
        self.riders[rider_id] = rider
        return rider

    def register_driver(
        self, driver_id: str, name: str, vehicle: Vehicle, location: Location
    ) -> Driver:
        driver = Driver(driver_id, name, vehicle, location)
        self.drivers[driver_id] = driver
        return driver

    def request_ride(
        self, rider_id: str, pickup: Location, drop: Location, vehicle_type: str
    ) -> Trip:
        rider = self.riders.get(rider_id)
        if not rider:
            raise ValueError(f"Rider {rider_id} not found.")

        available_drivers = list(self.drivers.values())
        matched_driver = self.matching_strategy.match_driver(
            pickup, available_drivers, vehicle_type
        )

        if not matched_driver:
            raise RuntimeError("No available drivers matched your request.")

        distance = pickup.distance_to(drop)
        fare = self.pricing_strategy.calculate_fare(
            distance, matched_driver.vehicle)

        self._trip_counter += 1
        trip_id = f"TRIP_{self._trip_counter:03d}"

        matched_driver.status = DriverStatus.ON_TRIP
        trip = Trip(trip_id, rider, matched_driver, pickup, drop, fare)
        trip.status = TripStatus.ACCEPTED

        self.trips[trip_id] = trip
        return trip


if __name__ == "__main__":
    service = RideServiceManager(
        matching_strategy=NearestDriverStrategy(),
        pricing_strategy=BasePricingStrategy(base_fare=5.0, rate_per_km=2.0),
    )

    alice = service.register_rider("R1", "Alice", "555-0192")

    d1 = service.register_driver(
        "D1", "Bob", SedanVehicle("CAB-101"), Location(1.0, 1.0)
    )
    d2 = service.register_driver(
        "D2", "Charlie", SedanVehicle("CAB-102"), Location(10.0, 10.0)
    )

    print(f"Registered Rider: {alice.name}")
    print(f"Registered Driver 1: {d1.name} at (1.0, 1.0)")
    print(f"Registered Driver 2: {d2.name} at (10.0, 10.0)")

    pickup_loc = Location(0.0, 0.0)
    drop_loc = Location(3.0, 4.0)

    trip = service.request_ride(
        rider_id="R1",
        pickup=pickup_loc,
        drop=drop_loc,
        vehicle_type="SedanVehicle"
    )

    print(f"\n--- Trip Created: {trip.trip_id} ---")
    print(f"Matched Driver: {trip.driver.name}")
    print(f"Calculated Fare: ${trip.fare:.2f}")
    print(f"Trip Status: {trip.status.name}")

    trip.start_trip()
    print(f"Trip Status after start: {trip.status.name}")

    trip.complete_trip()
    print(f"Trip Status after completion: {trip.status.name}")
    print(f"Driver {trip.driver.name} is now: {trip.driver.status.name}")

    trip.driver.add_rating(4.5)
    print(f"Updated Driver Rating: {trip.driver.rating}")
