-- clean view
IF OBJECT_ID('dbo.avg_speed_per_bin', 'V') IS NOT NULL
    DROP VIEW dbo.avg_speed_per_bin;
GO;

--- create view
CREATE VIEW dbo.avg_speed_per_bin
AS
WITH segment_first AS (
    SELECT
        t.segment_id,
        t.total_frames
    FROM (
        SELECT
            segment_id,
            total_frames,
            ROW_NUMBER() OVER (PARTITION BY segment_id ORDER BY id) AS rn
        FROM dbo.vehicle_speeds
    ) t
    WHERE t.rn = 1
),
frames_before AS (
    SELECT
        segment_id,
        SUM(total_frames) OVER (ORDER BY segment_id) - total_frames AS frames_before_segment
    FROM segment_first
),
enriched AS (
    SELECT
        v.carriageway,
        v.vehicle_type,
        v.speed_kmh,
        CAST(f.frames_before_segment + v.entry_frame AS float) / 25.0 AS timestamp_seconds
    FROM dbo.vehicle_speeds v
    INNER JOIN frames_before f
        ON v.segment_id = f.segment_id
),
binned AS (
    SELECT
        carriageway,
        vehicle_type,
        speed_kmh,
        FLOOR(timestamp_seconds / 300.0) AS bin_idx
    FROM enriched
)
SELECT
    CONCAT(
        CAST(CAST(bin_idx * 5 AS int) AS varchar(20)),
        '-',
        CAST(CAST(bin_idx * 5 + 5 AS int) AS varchar(20)),
        'min'
    ) AS time_bin,
    carriageway,
    vehicle_type,
    AVG(CAST(speed_kmh AS float)) AS avg_speed_kmh,
    COUNT(*) AS vehicle_count
FROM binned
GROUP BY
    bin_idx,
    carriageway,
    vehicle_type;
GO;

-- check view
select * from avg_speed_per_bin;
