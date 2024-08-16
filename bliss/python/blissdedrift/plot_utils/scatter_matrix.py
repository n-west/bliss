import altair as alt


def scatter_matrix_hits(list_of_hit_dicts):
    '''Plot hits as a matrix of scatterplots with histogram filters
    '''    
    data = alt.Data(values=list_of_hit_dicts)
    base = alt.Chart(data)

    brush = alt.selection_interval(encodings=['x'])
    scatter_select = alt.selection_interval(resolve='global')

    # Scatter plot
    points = base.mark_point(filled=True, color="black").encode(
        alt.X(alt.repeat("column"), type='quantitative').scale(zero=False),
        alt.Y(alt.repeat("row"), type='quantitative').scale(zero=False),
        color="origin:N"
        # color=alt.condition(scatter_select, 'origin:N', alt.value('lightgray'))
    ).properties(
        width=175,
        height=175
    ).repeat(
        row=['start_freq_MHz', 'drift_rate_Hz_per_sec', 'SNR', 'bandwidth_Hz'],
        column=['bandwidth_Hz', 'SNR', 'drift_rate_Hz_per_sec', 'start_freq_MHz']
    ).transform_filter(
        brush
    )
    # This is desirable, but causes weird interactivity issues
    # .add_params(
    #     scatter_select
    # )

    # Histograms
    snr_hist = alt.Chart(data).mark_bar().encode(
        alt.X("SNR", type="quantitative").bin(),
        y = "count()",
        color=alt.condition(brush, alt.value("black"), alt.value("lightgray"))
    ).properties(
        width=450,
        height=100
    ).add_selection(
        brush
    )

    bandwidth_Hz_hist = alt.Chart(data).mark_bar().encode(
        alt.X("bandwidth_Hz", type="quantitative").bin(),
        y = "count()",
        color=alt.condition(brush, alt.value("black"), alt.value("lightgray"))
    ).properties(
        width=450,
        height=100
    ).add_selection(
        brush
    )

    start_freq_MHz_hist = alt.Chart(data).mark_bar().encode(
        alt.X("start_freq_MHz", type="quantitative").bin(),
        y = "count()",
        color=alt.condition(brush, alt.value("black"), alt.value("lightgray"))
    ).properties(
        width=450,
        height=100
    ).add_selection(
        brush
    )

    bandwidth_Hz_hist = alt.Chart(data).mark_bar().encode(
        alt.X("bandwidth_Hz", type="quantitative").bin(),
        y = "count()",
        color=alt.condition(brush, alt.value("black"), alt.value("lightgray"))
    ).properties(
        width=450,
        height=100
    ).add_selection(
        brush
    )

    drift_rate_Hz_per_sec_hist = alt.Chart(data).mark_bar().encode(
        alt.X("drift_rate_Hz_per_sec", type="quantitative").bin(),
        y = "count()",
        color=alt.condition(brush, alt.value("black"), alt.value("lightgray"))
    ).properties(
        width=450,
        height=100
    ).add_selection(
        brush
    )

    chart = points & (snr_hist | bandwidth_Hz_hist) & (start_freq_MHz_hist | drift_rate_Hz_per_sec_hist)
    return chart