"""Source proposal derived from accepted host records, not a runtime admission."""
from collections import Counter,defaultdict
from runtime_plan import host_bytes
from weight_upload import schedule

def proposal(copies,dense_history,layer3_history):
    buckets=defaultdict(lambda:dict(copies=0,host_bytes=0));categories=Counter();diagnostics=Counter()
    def add(item,direction,count):
        size=host_bytes(item);bucket='small_le4096' if size<=4096 else 'medium_le65536' if size<=65536 else 'large_gt65536'
        row=buckets[direction+':'+bucket];row['copies']+=count;row['host_bytes']+=size*count
        categories[item['symbol']]+=count
    for family,h2d,d2h in [('uploads',1,2),('state_copies',0,8),('persistent_copies',0,2),
                           ('transport_copies',0,6),('diagnostic_copies',0,3)]:
        for item in copies[family]:
            add(item,'h2d',h2d);add(item,'d2h',d2h)
            if family=='diagnostic_copies':diagnostics[item['symbol']]+=1
    add(copies['hidden_input'],'h2d',3)
    add(copies['observer'],'d2h',3*copies['budget']['max_observer_polls_per_serial'])
    prior={g['direction']+':'+g['size_bucket']:g for g in dense_history['groups']}
    estimates={};mean=p95=rpc=0.
    for key,group in buckets.items():
        direction,bucket=key.split(':');basis=prior.get(key,prior[direction+':small_le4096'])
        # Only the large group scales by bytes. Per-call small-copy overhead
        # must not be inferred from a bulk-transfer average. Layer3 broad
        # >4KiB RPC statistics are a second, explicitly coarse cross-check.
        factor=(group['host_bytes']/basis['host_bytes'] if bucket=='large_gt65536' else group['copies']/basis['RPC']['count'])
        mean_part=basis['whole_copy_window']['sum_seconds']*factor
        p95_part=basis['whole_copy_window']['p95_seconds']*basis['RPC']['count']*factor
        rpc_part=basis['RPC']['sum_seconds']*factor
        estimates[key]=dict(**group,proxy_RPC_seconds=rpc_part,proxy_copy_window_mean_seconds=mean_part,
            proxy_copy_window_p95_scaled_seconds=p95_part,basis_direction=basis['direction'],basis_bucket=basis['size_bucket'])
        mean+=mean_part;p95+=p95_part;rpc+=rpc_part
    coarse=0.
    for key,g in buckets.items():
        direction,bucket=key.split(':');old=layer3_history['groups'][direction+(':'+'>4096B' if bucket!='small_le4096' else ':<=4096B')]
        coarse+=g['copies']*old['total']/old['count']
    profile=dict(startup_seconds=420,startup_pre_start_seconds=240,
        startup_device_seconds=180,capture_seconds=600,normal_stop_seconds=30,runtime_seconds=1080,
        compute_observer_seconds=90,observer_poll_interval_seconds=.5,capture_progress_seconds=15,
        parent_poll_seconds=.1,cpu=[0],address_space_bytes=4<<30,RSS_bytes=1<<30,guard_RSS_bytes=256<<20,swap_bytes=0,
        max_journal_bytes=128<<20,max_events=130000,capture_directory_bytes=384<<20,
        prepared_directory_bytes=1<<30,prepared_cache_bytes=4<<20,
        max_raw_bytes=copies['budget']['raw_byte_ceiling'],max_raw_files=copies['budget']['raw_file_ceiling'])
    groups=schedule([item['prepared'] for item in copies['uploads'] if item['symbol']=='weights'])
    profile.update(weight_upload=dict(channels=16,maximum_tasks=4,maximum_retained_bytes=64<<20,
                   maximum_copy_bytes=16<<20,batch_seconds=30),weight_upload_batches=len(groups))
    return dict(status='source_profile_proposal_not_admitted',profile=profile,
        budget=copies['budget'],copy_groups=dict(buckets),top10_total_copies=categories.most_common(10),
        top10_diagnostics_per_serial=diagnostics.most_common(10),estimates=estimates,
        proxy_total_RPC_seconds=rpc,proxy_total_copy_window_mean_seconds=mean,
        proxy_total_copy_window_p95_scaled_seconds=p95,Layer3_coarse_mean_RPC_crosscheck_seconds=coarse,
        estimate_scope='Historical host API/copy windows; scaled p95 values are not a prediction percentile, guarantee, pure wafer timing or measured Layer0 speed.',
        capture_allocation=dict(host_copy_and_durability_seconds=300,three_unmeasured_compute_windows_seconds=270,
            orchestration_margin_seconds=30),
        blocking_policy='Startup has nonrenewable240s pre-start preparation/management/upload and180s device-start windows, capped by420s overall. Each compute launch and its observer polls share one nonrenewable90s absolute window. Each bounded weight batch has one nonrenewable30s window.15s idle applies outside either active window;600s capture and1080s total deadlines always apply. Batch windows never extend the overall capture.',
        timing_separation=['SDK startup','original parameter H2D','retention D2H','raw diagnostic D2H',
            'journal/raw fsync and host checks','per-call atomic progress file and directory fsync',
            'submission to durable observer including polls','normal stop'],
        extra_progress_cost=dict(updates=copies['budget']['max_copies']+8,fsyncs_per_update=2,
            included_in_historical_copy_windows=False,measured_seconds=None,
            allocation='Part of300s host allocation; remaining margin beyond179.6s historical scaled copy windows also covers prepared-file reads, checks and progress durability.'),
        memory_accounting=dict(single_host_copy=copies['budget']['maximum_copy_host_bytes'],
            native_expected_and_result_and_serialized_buffers_at_most=5*(16<<20),
            asynchronous_retained_weights_at_most=64<<20,
            asynchronous_actual_schedule_peak=max(g['retained_host_bytes'] for g in groups),
            prepared_file_cache=4<<20,immutable_prepared_load_peak=3*((16<<20)+4096),
            metadata_binding_and_journal_indexes_reserved=128<<20,
            remaining_RSS_margin_for_SDK_and_interpreter='RSS1GiB ceiling requires parent enforcement; not a measured allocation'),
        raw_evidence=dict(maximum_bytes=copies['budget']['expected_raw_bytes'],
            maximum_files=copies['budget']['expected_raw_files'],arithmetic_bytes_per_serial=49592112,
            arithmetic_copies_per_serial=len(copies['diagnostic_copies']),
            omitted_duplicates='Successful resident parameters have immutable original prepared-file pins and exact bit-equality readback receipts; full raw kept on mismatch.',
            no_arithmetic_state_or_transport_subset=True),
        hardware_runs=0,no_budget_extension_or_automatic_retry=True)
