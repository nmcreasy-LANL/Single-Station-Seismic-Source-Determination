def pick_times_multicomponent(t, waveform_z, waveform_n, waveform_e, 
                               phases=["P", "S"], 
                               distance_range=None, 
                               depth_range=None,
                               event_time=0,
                               start_time=None):
    """
    Interactive phase picker for three-component seismic data with TauP predictions.
    
    Parameters:
    -----------
    t : array-like
        Time array in seconds (relative to trace start)
    waveform_z : array-like
        Vertical component waveform data
    waveform_n : array-like
        North component waveform data
    waveform_e : array-like
        East component waveform data
    phases : list of str
        List of phase names to pick (default: ["P", "S"])
    distance_range : list [min, max]
        Distance range in degrees for TauP predictions (required for TauP)
    depth_range : list [min, max]
        Depth range in km for TauP predictions (required for TauP)
    event_time : float
        Reference event time in seconds (default: 0)
    start_time : obspy.UTCDateTime, optional
        Start time of the trace for UTC display. If None, uses relative time.
    
    Returns:
    --------
    picks : dict
        Dictionary with phase names as keys and lists of pick times as values
        (times are in seconds relative to trace start, regardless of display mode)
    
    Controls:
    ---------
    Mouse:
        - Left click: pick current phase on any trace
        - Right drag: zoom in (synchronized across all traces)
        - Single right click: reset zoom
    Keyboard:
        - Number keys (1-9): switch between phases
        - 'u': undo last pick
        - 't': toggle TauP prediction visibility
        - 'q': quit and return picks
    """
    
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import matplotlib.dates as mdates
    from matplotlib.dates import DateFormatter, SecondLocator
    
    # Try to import obspy for TauP
    taup_available = False
    taup_model = None
    if distance_range is not None and depth_range is not None:
        try:
            from obspy.taup import TauPyModel
            taup_model = TauPyModel(model="iasp91")
            taup_available = True
        except ImportError:
            print("Warning: obspy not available. TauP predictions disabled.")
    
    mpl.rcParams['keymap.save'] = []
    
    # Initialize picks dictionary
    picks = {phase: [] for phase in phases}
    pick_lines = []
    current_phase = phases[0]
    
    # Convert time to UTC if start_time provided
    use_utc = (start_time is not None)
    if use_utc:
        # Convert time array to UTC datetime objects and then to matplotlib date numbers
        t_utc = [start_time + ti for ti in t]
        t_plot = mdates.date2num(t_utc)
        print(f"  Using UTC time display. Start time: {start_time}")
    else:
        t_plot = t
        print(f"  Using relative time display (seconds)")
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 3.5), sharex=True)
    axes = [ax1, ax2, ax3]
    
    # DEBUG: Verify data received by picker
    #print(f"\nDEBUG in pick_times_multicomponent:")
    #print(f"  waveform_z max amplitude: {np.max(np.abs(waveform_z)):.2e}")
    #print(f"  waveform_n max amplitude: {np.max(np.abs(waveform_n)):.2e}")
    #print(f"  waveform_e max amplitude: {np.max(np.abs(waveform_e)):.2e}")
    
    # Normalize waveforms for display
    norm_z = waveform_z / np.max(np.abs(waveform_z))
    norm_n = waveform_n / np.max(np.abs(waveform_n))
    norm_e = waveform_e / np.max(np.abs(waveform_e))
    
    # Plot waveforms
    ax1.plot(t_plot, norm_z, color="black", lw=0.8)
    ax1.set_ylabel("Z (Vertical)", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=7)
    
    ax2.plot(t_plot, norm_n, color="black", lw=0.8)
    ax2.set_ylabel("N (North)", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=7)
    
    ax3.plot(t_plot, norm_e, color="black", lw=0.8)
    ax3.set_ylabel("E (East)", fontsize=8)
    if use_utc:
        ax3.set_xlabel("UTC Time (HH:MM:SS)", fontsize=8)
        # Format x-axis for UTC time
        ax3.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
        # Set appropriate tick interval based on data duration
        data_duration_sec = t[-1] - t[0] if len(t) > 1 else 3600
        if data_duration_sec > 3000:
            interval = 600  # 10 minutes
        elif data_duration_sec > 1800:
            interval = 300  # 5 minutes
        else:
            interval = 120  # 2 minutes
        ax3.xaxis.set_major_locator(SecondLocator(interval=interval))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax3.set_xlabel("Time (s)", fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=7)
    
    # Title showing current phase
    phase_list_str = ", ".join([f"{i+1}={p}" for i, p in enumerate(phases)])
    title = fig.suptitle(f"Picking phase: {current_phase} | Phases: {phase_list_str} | 'u'=undo 't'=toggle TauP 'q'=quit", 
                         fontsize=9, fontweight='bold')
    
    # Define colors for phases (cycle through if more than predefined)
    base_colors = ["red", "blue", "green", "orange", "purple", "brown", "magenta", "cyan", "olive"]
    phase_colors = {phases[i]: base_colors[i % len(base_colors)] for i in range(len(phases))}
    
    full_xlim = ax3.get_xlim()  # Store full waveform x-limits for unzoom
    
    # Set initial zoom to first 1800 seconds (30 minutes)
    if use_utc:
        # Find the plot coordinate for 1800 seconds
        idx_1800 = np.searchsorted(t, 1800)
        if idx_1800 < len(t_plot):
            ax3.set_xlim(t_plot[0], t_plot[idx_1800])
        else:
            ax3.set_xlim(t_plot[0], t_plot[-1])
    else:
        ax3.set_xlim(0, 1800)
    
    # Calculate optimal y-limits for first 1800 seconds
    idx_1800 = np.searchsorted(t, 1800)
    max_z_1800 = np.max(np.abs(norm_z[:idx_1800])) if idx_1800 > 0 else np.max(np.abs(norm_z))
    max_n_1800 = np.max(np.abs(norm_n[:idx_1800])) if idx_1800 > 0 else np.max(np.abs(norm_n))
    max_e_1800 = np.max(np.abs(norm_e[:idx_1800])) if idx_1800 > 0 else np.max(np.abs(norm_e))
    
    # Set y-limits with some padding (1.1x the max amplitude)
    ax1.set_ylim(-1.1 * max_z_1800, 1.1 * max_z_1800)
    ax2.set_ylim(-1.1 * max_n_1800, 1.1 * max_n_1800)
    ax3.set_ylim(-1.1 * max_e_1800, 1.1 * max_e_1800)
    
    # TauP predictions storage
    taup_patches = []
    taup_visible = True
    
    def calculate_taup_predictions():
        """Calculate TauP arrival time ranges for all phases."""
        if not taup_available or taup_model is None:
            return {}
        
        predictions = {}
        
        for phase in phases:
            try:
                # Calculate arrivals for distance and depth ranges
                dist_min, dist_max = distance_range
                depth_min, depth_max = depth_range
                
                # Get arrivals at corner points
                arrivals_list = []
                for dist in [dist_min, dist_max]:
                    for depth in [depth_min, depth_max]:
                        try:
                            arrivals = taup_model.get_travel_times(
                                source_depth_in_km=depth,
                                distance_in_degree=dist,
                                phase_list=[phase]
                            )
                            if arrivals:
                                arrivals_list.extend([a.time for a in arrivals])
                        except:
                            pass
                
                if arrivals_list:
                    # TauP times are relative to event origin (t=0 at event)
                    # event_time = waveform_start - event_origin (diff1)
                    # Waveform time axis t starts at 0 (waveform start)
                    #
                    # If event_time > 0: waveform starts AFTER event, event is at t = -event_time (negative!)
                    # If event_time < 0: waveform starts BEFORE event, event is at t = -event_time (positive!)
                    # If event_time ≈ 0: waveform starts AT event, event is at t ≈ 0
                    #
                    # Phase arrives at: (time of event on waveform axis) + (travel time from event)
                    #                 = -event_time + taup_time
                    taup_min = min(arrivals_list)
                    taup_max = max(arrivals_list)
                    waveform_min = -event_time + taup_min
                    waveform_max = -event_time + taup_max
                    
                    print(f"  TauP {phase}: {taup_min:.1f}-{taup_max:.1f}s from event, " +
                          f"waveform coords: {waveform_min:.1f}-{waveform_max:.1f}s " +
                          f"(event_time={event_time:.2f}, event at t={-event_time:.2f})")
                    
                    predictions[phase] = {
                        'min': waveform_min,
                        'max': waveform_max
                    }
            except:
                pass
        
        return predictions
    
    def plot_taup_predictions():
        """Plot TauP prediction ranges on all axes."""
        # Clear existing patches
        for patch_set in taup_patches:
            for patch in patch_set:
                patch.remove()
        taup_patches.clear()
        
        if not taup_visible or not taup_available:
            fig.canvas.draw_idle()
            return
        
        predictions = calculate_taup_predictions()
        
        for phase, times in predictions.items():
            color = phase_colors.get(phase, "gray")
            
            # Convert prediction times to plot coordinates (UTC if needed)
            if use_utc:
                # times['min'] and times['max'] are in relative seconds
                # Convert to UTC plot coordinates
                tmin_plot = mdates.date2num(start_time + times['min'])
                tmax_plot = mdates.date2num(start_time + times['max'])
            else:
                tmin_plot = times['min']
                tmax_plot = times['max']
            
            for ax in axes:
                ylim = ax.get_ylim()
                # Create shaded region for predicted arrival range
                patch = Rectangle(
                    (tmin_plot, ylim[0]), 
                    tmax_plot - tmin_plot, 
                    ylim[1] - ylim[0],
                    linewidth=0, 
                    facecolor=color, 
                    alpha=0.15,
                    zorder=0
                )
                ax.add_patch(patch)
                
                # Add text label at top
                text = ax.text(
                    (tmin_plot + tmax_plot) / 2, 
                    ylim[1] * 0.9,
                    phase + " (TauP)",
                    ha='center', 
                    va='top',
                    fontsize=8,
                    color=color,
                    alpha=0.7,
                    zorder=1
                )
                taup_patches.append([patch, text])
        
        fig.canvas.draw_idle()
    
    # Initial TauP plot
    if taup_available:
        plot_taup_predictions()
    
    # Zoom state
    zoom_start = {"x": None}
    zoom_rect = None
    
    def print_picks():
        print("\n" + "="*60)
        print("PICKED PHASES:")
        print("="*60)
        for phase in phases:
            if picks[phase]:
                for i, tt in enumerate(picks[phase], 1):
                    print(f"  {phase}{i}: {tt:.4f} s")
        print("="*60)
    
    # Mouse event handlers
    def onpress(event):
        nonlocal zoom_rect
        
        # Find which axis was clicked
        clicked_ax = None
        for ax in axes:
            if event.inaxes == ax:
                clicked_ax = ax
                break
        
        if clicked_ax is None:
            return
        
        if event.button == 3:  # Right click - start zoom
            zoom_start["x"] = event.xdata
            # Create zoom rectangles on all axes
            zoom_rect = []
            for ax in axes:
                ylim = ax.get_ylim()
                rect = Rectangle(
                    (event.xdata, ylim[0]), 
                    0, 
                    ylim[1] - ylim[0],
                    linewidth=1, 
                    edgecolor='blue', 
                    facecolor='blue', 
                    alpha=0.2
                )
                ax.add_patch(rect)
                zoom_rect.append(rect)
            fig.canvas.draw_idle()
        
        elif event.button == 1:  # Left click - pick
            # Convert click position to relative time (for finding correct data index)
            if use_utc:
                # event.xdata is matplotlib date number, convert to relative seconds
                # Find closest point in t_plot array
                idx = np.argmin(np.abs(t_plot - event.xdata))
                pick_t = t[idx]  # Store as relative time
                pick_plot = t_plot[idx]  # Plot at UTC position
            else:
                # Already in relative time
                idx = np.argmin(np.abs(t - event.xdata))
                pick_t = t[idx]
                pick_plot = pick_t
            
            picks[current_phase].append(pick_t)
            
            # Draw pick lines on all three axes
            lines = []
            for ax in axes:
                line = ax.axvline(
                    pick_plot,  # Use plot coordinates (UTC or relative)
                    color=phase_colors[current_phase], 
                    linestyle="-",
                    linewidth=1,
                    zorder=10
                )
                lines.append(line)
            
            pick_lines.append((current_phase, lines))
            if use_utc:
                utc_time = start_time + pick_t
                print(f"{current_phase} pick: {pick_t:.4f} s (UTC: {utc_time})")
            else:
                print(f"{current_phase} pick: {pick_t:.4f} s")
            fig.canvas.draw_idle()
    
    def onmotion(event):
        if zoom_rect is None:
            return
        
        # Update all zoom rectangles
        x0 = zoom_start["x"]
        if x0 is not None and event.xdata is not None:
            width = event.xdata - x0
            for rect in zoom_rect:
                rect.set_width(width)
            fig.canvas.draw_idle()
    
    def onrelease(event):
        nonlocal zoom_rect
        
        if event.button != 3:
            return
        
        x0 = zoom_start["x"]
        x1 = event.xdata
        zoom_start["x"] = None
        
        if x0 is None or x1 is None:
            if zoom_rect:
                for rect in zoom_rect:
                    rect.remove()
                zoom_rect = None
            fig.canvas.draw_idle()
            return
        
        if abs(x1 - x0) < 1e-3:  # Single click - reset zoom
            ax3.set_xlim(full_xlim)
        else:  # Drag - apply zoom
            ax3.set_xlim(min(x0, x1), max(x0, x1))
        
        # Remove zoom rectangles
        if zoom_rect:
            for rect in zoom_rect:
                rect.remove()
            zoom_rect = None
        
        # Update TauP patches after zoom
        if taup_available:
            plot_taup_predictions()
        
        fig.canvas.draw_idle()
    
    # Keyboard event handler
    def onkey(event):
        nonlocal current_phase, taup_visible
        
        # Number keys to switch phases
        if event.key in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
            idx = int(event.key) - 1
            if idx < len(phases):
                current_phase = phases[idx]
                phase_list_str = ", ".join([f"{i+1}={p}" for i, p in enumerate(phases)])
                title.set_text(f"Picking phase: {current_phase} | Phases: {phase_list_str} | 'u'=undo 't'=toggle TauP 'q'=quit")
        
        # Undo last pick
        elif event.key == "u" and pick_lines:
            phase, lines = pick_lines.pop()
            for line in lines:
                line.remove()
            removed = picks[phase].pop()
            print(f"Removed {phase} pick: {removed:.4f} s")
        
        # Toggle TauP visibility
        elif event.key == "t":
            taup_visible = not taup_visible
            plot_taup_predictions()
            status = "ON" if taup_visible else "OFF"
            print(f"TauP predictions: {status}")
        
        # Quit
        elif event.key == "q":
            print_picks()
            plt.close(fig)
        
        fig.canvas.draw_idle()
    
    # Connect event handlers
    fig.canvas.mpl_connect("button_press_event", onpress)
    fig.canvas.mpl_connect("motion_notify_event", onmotion)
    fig.canvas.mpl_connect("button_release_event", onrelease)
    fig.canvas.mpl_connect("key_press_event", onkey)
    
    plt.tight_layout()
    plt.show()  # Blocks until closed
    
    print_picks()  # Final print
    return picks
