"""Validate bundled sensors, or extract a new sensor realization from local flow_full70.p.
The historical sensor perturbation seed was not saved. Use the bundled data for paper tables.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from evaluate import load_p

ROOT = Path(__file__).resolve().parents[2]
SENSORS = np.array([[62,12],[52,44],[77,15],[43,35],[74,35],[109,40],[125,15],[80,35],[100,20]])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, help='Local trusted flow_full70.p; omitted to validate bundled data.')
    p.add_argument('--seed', type=int, default=0, help='Seed for a NEW spatial-noise realization.')
    p.add_argument('--output-dir', type=Path, default=ROOT/'data/cylinder')
    args = p.parse_args()
    if args.raw is None:
        path = args.output_dir/'flow_sensors.p'
        data = load_p(path)
        assert data[0].shape == (200,9) and data[1].shape == (15000,9)
        assert data[2].shape == data[1].shape
        assert data[4] == .01 and data[6] == .1
        assert all(np.isfinite(x).all() for x in data[:3])
        print(json.dumps({'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                          'train_shape': list(data[0].shape), 'test_shape': list(data[1].shape),
                          'dt_raw': data[4], 'dt_observed': data[6]}, indent=2))
        return
    output = args.output_dir/'flow_sensors.p'
    if output.exists():
        raise FileExistsError(f'Refusing to replace {output}; choose a new --output-dir.')
    raw = load_p(args.raw)
    # Original Flow_data_preparison.py: discard 3000 frames, scale lattice speed by 100.
    frames = np.asarray(raw[0])[3000:]
    if frames.ndim != 3 or frames.shape[1:] != (150,50) or len(frames) <= 2000:
        raise ValueError('Expected raw frame array [time,150,50] with >5000 frames.')
    rate, n_train, dx, dv = 10, 200, .1, 100
    dt = (dx/dv) * int(raw[4])
    error = .5 * dx
    observed = frames[:,SENSORS[:,0],SENSORS[:,1]].astype(np.float64) * dv
    noised = np.zeros_like(observed)
    rng = np.random.RandomState(args.seed)
    grid = (np.arange(150)*dx, np.arange(50)*dx)
    # Keep original sensor-major RNG order and bilinear interpolation.
    for i, sensor in enumerate(SENSORS):
        for t in range(len(frames)):
            position = rng.normal(sensor*dx, error)
            interp = RegularGridInterpolator(grid, frames[t]*dv)
            noised[t,i] = interp(position)[0]
        print(f'Extracted sensor {i+1}/{len(SENSORS)}', flush=True)
    data = [noised[::rate][:n_train], observed[n_train*rate:], noised[n_train*rate:],
            rate, dt, error, dt*rate, dx]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as f:
        pickle.dump(data, f)
    (args.output_dir/'preparation_manifest.json').write_text(json.dumps(
        {'source': str(args.raw), 'noise_seed': args.seed, 'historical_noise_seed': 'unknown',
         'paper_exact_realization': False, 'sensor_indices': SENSORS.tolist(),
         'dropped_frames': 3000, 'train_count': 200, 'sample_rate': 10,
         'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2)+'\n')
    print(f'Saved new sensor realization to {output}')


if __name__ == '__main__':
    main()
