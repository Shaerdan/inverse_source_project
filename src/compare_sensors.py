#!/usr/bin/env python3
"""
Compare sensor locations between CLI and my diagnostic.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from mesh import get_polygon_sensor_locations


def my_generate_square_sensors(n_sensors=100):
    """My manual implementation."""
    perimeter = 8.0
    sensor_locations = []
    for i in range(n_sensors):
        t = i * perimeter / n_sensors
        if t < 2:
            sensor_locations.append((-1 + t, -1))
        elif t < 4:
            sensor_locations.append((1, -1 + (t - 2)))
        elif t < 6:
            sensor_locations.append((1 - (t - 4), 1))
        else:
            sensor_locations.append((-1, 1 - (t - 6)))
    return np.array(sensor_locations)


def main():
    print("="*70)
    print("COMPARING SENSOR LOCATIONS: CLI vs My Diagnostic")
    print("="*70)
    
    vertices = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    
    # CLI method
    cli_sensors = get_polygon_sensor_locations(vertices, 100)
    
    # My method
    my_sensors = my_generate_square_sensors(100)
    
    print(f"\nCLI sensors (get_polygon_sensor_locations):")
    print(f"  Shape: {cli_sensors.shape}")
    print(f"  First 5: {cli_sensors[:5]}")
    print(f"  Last 5: {cli_sensors[-5:]}")
    
    print(f"\nMy sensors (manual):")
    print(f"  Shape: {my_sensors.shape}")
    print(f"  First 5: {my_sensors[:5]}")
    print(f"  Last 5: {my_sensors[-5:]}")
    
    print(f"\nComparison:")
    if cli_sensors.shape == my_sensors.shape:
        max_diff = np.max(np.abs(cli_sensors - my_sensors))
        print(f"  Max difference: {max_diff:.6e}")
        print(f"  Are they close? {np.allclose(cli_sensors, my_sensors)}")
        
        if not np.allclose(cli_sensors, my_sensors):
            print(f"\n  ⚠️ SENSORS ARE DIFFERENT!")
            # Find where they differ
            for i in range(min(10, len(cli_sensors))):
                diff = np.linalg.norm(cli_sensors[i] - my_sensors[i])
                if diff > 1e-10:
                    print(f"    Sensor {i}: CLI={cli_sensors[i]}, Mine={my_sensors[i]}, diff={diff:.6e}")
    else:
        print(f"  ⚠️ SHAPES DON'T MATCH!")


if __name__ == '__main__':
    main()
