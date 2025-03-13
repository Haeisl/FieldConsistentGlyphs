def print_metrics(start_time, end_time, glyph_start_time, glyph_creation_times, points, lines):
        print(f"{'#'*35} METRICS: {'#'*35}")
        print(f"Glyph No. | Time To Compute | Origin")
        glyph_processing_times = []
        glyph_cum_dur = 0
        for idx, point, glyph_end_time in glyph_creation_times:
            if isinstance(glyph_end_time, str):
                print(f"Glyph {idx+1 if idx >=10 else '0'+str(idx+1)}  | {glyph_end_time}| {[round(p, 3) for p in point.tolist()[:2]]} ")
                continue
            glyph_duration = glyph_end_time - glyph_start_time  - glyph_cum_dur
            glyph_processing_times.append(glyph_duration)
            print(f"Glyph {idx if idx >=10 else '0'+str(idx)}  | {glyph_duration:.2f}s{' '*11}| {[round(p, 3) for p in point.tolist()[:2]]} ")
            glyph_cum_dur += glyph_duration
        print("")
        print(f"Total Duration: {end_time - start_time:.2f} s")
        print(f"Setup Took: {glyph_start_time - start_time}s")
        print(f"Average Processing Time Per Glyph: {sum([float(t) for t in glyph_processing_times])/len(glyph_processing_times):.2f}")
        print(f"There were {points.GetNumberOfPoints()} Points drawn in {lines.GetNumberOfCells()} Lines")
        print(f"{'#'*80}")