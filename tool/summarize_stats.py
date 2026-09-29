#!/usr/bin/env python3
"""Summarise a cs_run stats.csv the way Zheng et al. 2025 Table 2 reports it.

Usage: summarize_stats.py <stats.csv> [--paper fig4]

Table 2 columns: #CG is the average (peak) CG iterations *per linear solve*, #Newton the
average (peak) Newton iterations *per time step*, #contact the average (peak) number of
contact pairs, and time the average (peak) runtime per time step. The component breakdown
(Hess / PCG / CCD / LS / misc) is the total seconds spent in each stage over the run,
divided by the number of frames.
"""
import csv, sys, statistics as st

# (name, #CG avg, #CG peak, #Newton avg, #Newton peak, contacts avg, contacts peak,
#  TOI, t_avg, t_peak, hess, pcg, ccd, ls, misc)
PAPER = {
    'fig4': dict(cg=(53.59, 669), newton=(19.23, 37), contacts=(37.9e3, 102.6e3), toi=0.242,
                 time=(1.300, 4.11), hess=0.237, pcg=0.534, ccd=0.446, ls=0.051, misc=0.033,
                 label='Zheng et al. 2025, Fig. 4 (animal well), RTX 4090'),
    'fig21': dict(cg=(44.65, 238), newton=(11.80, 34), contacts=(27.0e3, 187.4e3), toi=0.319,
                  time=(0.580, 3.57), hess=0.107, pcg=0.212, ccd=0.216, ls=0.019, misc=0.026,
                  label='Zheng et al. 2025, Fig. 21 (trapped squishy balls), RTX 4090'),
    'fig1': dict(cg=(28.35, 146), newton=(30.09, 47), contacts=(0.53e6, 1.45e6), toi=0.192,
                 time=(5.367, 12.39), hess=0.762, pcg=1.229, ccd=3.124, ls=0.134, misc=0.118,
                 label='Zheng et al. 2025, Fig. 1 (squishy balls under extreme compression), RTX 4090'),
    # The rows below are the statistics table only (p. 16): no TOI and no stage breakdown.
    'fig3': dict(cg=(67.1, 773), newton=(16.1, 103), contacts=(42.6e3, 542.8e3), toi=None,
                 time=(0.66, 56.86), hess=None, pcg=None, ccd=None, ls=None, misc=None,
                 label='Zheng et al. 2025, Fig. 3 (compressing chain rings), RTX 4090'),
    'fig5': dict(cg=(73.0, 179), newton=(20.8, 51), contacts=(5.4e3, 26.6e3), toi=None,
                 time=(0.33, 0.97), hess=None, pcg=None, ccd=None, ls=None, misc=None,
                 label='Zheng et al. 2025, Fig. 5 (ramen), RTX 4090'),
    'fig6': dict(cg=(82.5, 1520), newton=(13.4, 31), contacts=(4.7e3, 6.7e3), toi=None,
                 time=(0.18, 0.45), hess=None, pcg=None, ccd=None, ls=None, misc=None,
                 label='Zheng et al. 2025, Fig. 6 (twisting rods), RTX 4090'),
    'fig8': dict(cg=(33.6, 619), newton=(6.0, 6), contacts=(35.8, 803), toi=None,
                 time=(0.16, 0.17), hess=None, pcg=None, ccd=None, ls=None, misc=None,
                 label='Zheng et al. 2025, Fig. 8 (momentum conservation), RTX 4090'),
    'fig13': dict(cg=(83.7, 280), newton=(7.6, 195), contacts=(8.9e3, 31.3e3), toi=None,
                  time=(0.17, 7.60), hess=None, pcg=None, ccd=None, ls=None, misc=None,
                  label='Zheng et al. 2025, Fig. 13 (pig falling), RTX 4090'),
    'fig14': dict(cg=(143.0, 719), newton=(9.4, 29), contacts=(1.2e3, 5.1e3), toi=None,
                  time=(0.15, 0.45), hess=None, pcg=None, ccd=None, ls=None, misc=None,
                  label='Zheng et al. 2025, Fig. 14 (friction roller), RTX 4090'),
    'fig16': dict(cg=(71.5, 1134), newton=(8.1, 36), contacts=(1.3e3, 9.4e3), toi=None,
                  time=(0.11, 1.58), hess=None, pcg=None, ccd=None, ls=None, misc=None,
                  label='Zheng et al. 2025, Fig. 16 (dolphin and funnel), RTX 4090'),
}

def main():
    path = sys.argv[1]
    ref = None
    if '--paper' in sys.argv:
        ref = PAPER[sys.argv[sys.argv.index('--paper') + 1]]
    rows = list(csv.DictReader(open(path)))
    if not rows:
        sys.exit('no rows in ' + path)
    col = lambda k: [float(r[k]) for r in rows]
    n = len(rows)

    newton = col('inner_newton')
    solves = sum(newton)
    cg_tot = col('pcg_iters_total')
    cg_avg = sum(cg_tot) / solves if solves else 0.0
    cg_peak = max(col('pcg_iters_avg'))          # peak per-step average; see note below
    pairs = [a + b + c for a, b, c in zip(col('pairs_pt'), col('pairs_ee'), col('pairs_ph'))]
    t = col('time_ms')
    stage = {k: sum(col('ms_' + k)) / n / 1e3 for k in
             ('assemble', 'pcg', 'ccd', 'line_search', 'elem_cache', 'gradient', 'active_set', 'contact')}
    misc = sum(t) / n / 1e3 - (stage['assemble'] + stage['pcg'] + stage['ccd'] + stage['line_search'])
    # The paper's TOI column is the mean accepted step over CCD calls, so contact-free frames
    # (where alpha is trivially 1) must not dilute it.
    alpha = [a for a, p in zip(col('alpha_mean'), pairs) if p > 0]
    toi = st.mean(alpha) if alpha else 0.0
    pen = max(col('max_penetration'))

    if '--markdown' in sys.argv:
        r = ref or {}
        def row(name, ours, theirs, fmt='{:.3f}'):
            o = fmt.format(ours)
            t = fmt.format(theirs) if theirs is not None else '-'
            rat = '{:.2f}x'.format(ours / theirs) if theirs else '-'
            print('| {} | {} | {} | {} |'.format(name, o, t, rat))
        print('| quantity | ours | paper | ratio |')
        print('| --- | ---: | ---: | ---: |')
        row('#CG, average per linear solve', cg_avg, r.get('cg', (None,))[0], '{:.2f}')
        row('#Newton, average per time step', solves / n, r.get('newton', (None,))[0], '{:.2f}')
        row('#Newton, peak', max(newton), r.get('newton', (0, None))[1], '{:.0f}')
        row('#contact pairs, average', sum(pairs) / n, r.get('contacts', (None,))[0], '{:.0f}')
        row('#contact pairs, peak', max(pairs), r.get('contacts', (0, None))[1], '{:.0f}')
        row('TOI, mean accepted step', toi, r.get('toi'), '{:.3f}')
        row('time, average (s / step)', sum(t) / n / 1e3, r.get('time', (None,))[0])
        row('time, peak (s / step)', max(t) / 1e3, r.get('time', (0, None))[1])
        print('| | | | |')
        row('Hess, assembly (s / step)', stage['assemble'], r.get('hess'))
        row('PCG (s / step)', stage['pcg'], r.get('pcg'))
        row('CCD (s / step)', stage['ccd'], r.get('ccd'))
        row('line search (s / step)', stage['line_search'], r.get('ls'))
        row('misc (s / step)', misc, r.get('misc'))
        return

    def line(name, ours, theirs=None, fmt='{:.3f}'):
        o = fmt.format(ours)
        if theirs is None:
            print(f'  {name:<34}{o:>14}')
        else:
            r = fmt.format(theirs)
            print(f'  {name:<34}{o:>14}{r:>14}   {ours/theirs:5.2f}x' if theirs else
                  f'  {name:<34}{o:>14}{r:>14}')

    print(f'\n{path}: {n} frames, {int(solves)} linear solves\n')
    hdr = f'  {"":<34}{"ours":>14}{"paper":>14}' if ref else f'  {"":<34}{"ours":>14}'
    print(hdr)
    print('  ' + '-' * (62 if ref else 48))
    line('#CG  avg per solve', cg_avg, ref and ref['cg'][0], '{:.2f}')
    line('#CG  peak (per-step avg)', cg_peak, ref and ref['cg'][1], '{:.0f}')
    line('#Newton  avg per step', solves / n, ref and ref['newton'][0], '{:.2f}')
    line('#Newton  peak', max(newton), ref and ref['newton'][1], '{:.0f}')
    line('#contacts  avg', sum(pairs) / n, ref and ref['contacts'][0], '{:.0f}')
    line('#contacts  peak', max(pairs), ref and ref['contacts'][1], '{:.0f}')
    line('TOI (mean accepted alpha)', toi, ref and ref['toi'], '{:.3f}')
    line('time  avg (s/step)', sum(t) / n / 1e3, ref and ref['time'][0], '{:.3f}')
    line('time  peak (s/step)', max(t) / 1e3, ref and ref['time'][1], '{:.3f}')
    print('  ' + '-' * (62 if ref else 48))
    line('  Hess (assemble)', stage['assemble'], ref and ref['hess'])
    line('  PCG', stage['pcg'], ref and ref['pcg'])
    line('  CCD', stage['ccd'], ref and ref['ccd'])
    line('  LS', stage['line_search'], ref and ref['ls'])
    line('  misc', misc, ref and ref['misc'])
    print('  ' + '-' * (62 if ref else 48))
    print(f'  {"max penetration (m)":<34}{pen:>14.3e}')
    print(f'  {"frames that hit the iteration cap":<34}{int(sum(col("hit_cap"))):>14d}')
    if ref:
        print(f'\n  paper reference: {ref["label"]}')
    print('\n  note: the paper reports peak #CG per linear solve; cs_run records the per-step')
    print('  average, so our peak column is a lower bound on the true per-solve peak.\n')

main()
