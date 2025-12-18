-- Count all matching Agentic-PRs: 7,156 (4,963 Accepted, 2,193 Rejected)
-- 1. closed PRs
-- 2. from repository licensed under MIT or Apache-2.0
-- 3. has at least one non-creator review or comment

SELECT COUNT(*) AS matching_pull_requests
FROM pull_request pr
JOIN repository repo
  ON repo.id = pr.repo_id
WHERE pr.STATE = 'closed'
  -- accepted PRs, and remove NOT for rejected PRs
  -- AND pr.merged_at IS NOT NULL
  AND repo.license IN ('MIT', 'Apache-2.0')
  AND (
        -- at least one non-creator review
        EXISTS (
            SELECT 1
            FROM pr_reviews r
            WHERE r.pr_id = pr.id
              AND r.user <> pr.user
              AND CAST(r.submitted_at AS TIMESTAMP) < CAST(pr.closed_at AS TIMESTAMP)
        )
        OR
        -- at least one non-creator comment
        EXISTS (
            SELECT 1
            FROM pr_comments c
            WHERE c.pr_id = pr.id
              AND c.user <> pr.user
              AND CAST(c.created_at AS TIMESTAMP) < CAST(pr.closed_at AS TIMESTAMP)
        )
  );

-- Count the mean, median, and standard deviation of days to closure  
-- Accepted PRs: 3.006649, 1, 5.939937
-- Rejected PRs: 6.894209, 2, 11.113096

WITH matching_prs AS (
  SELECT
    pr.id,
    CEILING(
      date_diff(
        'second',
        CAST(pr.created_at AS TIMESTAMP),
        CAST(pr.closed_at AS TIMESTAMP)
      ) / 86400.0
    ) AS days_to_close
  FROM pull_request pr
  JOIN repository repo
    ON repo.id = pr.repo_id
  WHERE pr.state = 'closed'
    AND pr.merged_at IS NOT NULL -- Remove NOT for rejected PRs
    AND repo.license IN ('MIT', 'Apache-2.0')
    AND (
      EXISTS (
        SELECT 1
        FROM pr_reviews r
        WHERE r.pr_id = pr.id
          AND r.user <> pr.user
          AND CAST(r.submitted_at AS TIMESTAMP) < CAST(pr.closed_at AS TIMESTAMP)
      )
      OR
      EXISTS (
        SELECT 1
        FROM pr_comments c
        WHERE c.pr_id = pr.id
          AND c.user <> pr.user
          AND CAST(c.created_at AS TIMESTAMP) < CAST(pr.closed_at AS TIMESTAMP)
      )
    )
)
SELECT
  AVG(days_to_close) AS mean_days,
  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY days_to_close) AS median_days,
  STDDEV_POP(days_to_close) AS stddev_days
FROM matching_prs;

-- Count days_to_close, accepted_prs, rejected_prs, total_prs, proportion_accepted, cumulative_proportion_accepted

WITH matching_prs AS (
  SELECT
    pr.id,
    CEILING(
      date_diff(
        'second',
        CAST(pr.created_at AS TIMESTAMP),
        CAST(pr.closed_at  AS TIMESTAMP)
      ) / 86400.0
    ) AS days_to_close,
    CASE WHEN pr.merged_at IS NOT NULL THEN 1 ELSE 0 END AS is_accepted
  FROM pull_request pr
  JOIN repository repo
    ON repo.id = pr.repo_id
  WHERE pr.state = 'closed'
    AND repo.license IN ('MIT', 'Apache-2.0')
    AND (
      EXISTS (
        SELECT 1
        FROM pr_reviews r
        WHERE r.pr_id = pr.id
          AND r.user <> pr.user
          AND CAST(r.submitted_at AS TIMESTAMP) < CAST(pr.closed_at AS TIMESTAMP)
      )
      OR
      EXISTS (
        SELECT 1
        FROM pr_comments c
        WHERE c.pr_id = pr.id
          AND c.user <> pr.user
          AND CAST(c.created_at AS TIMESTAMP) < CAST(pr.closed_at AS TIMESTAMP)
      )
    )
),
daily AS (
  SELECT
    days_to_close,
    SUM(is_accepted) AS accepted_prs,
    SUM(1 - is_accepted) AS rejected_prs,
    COUNT(*) AS total_prs
  FROM matching_prs
  GROUP BY days_to_close
)
SELECT
  days_to_close,
  accepted_prs,
  rejected_prs,
  total_prs,
  accepted_prs * 1.0 / NULLIF(total_prs, 0) AS proportion_accepted,
  SUM(accepted_prs) OVER (ORDER BY days_to_close) * 1.0
    / NULLIF(SUM(total_prs) OVER (ORDER BY days_to_close), 0) AS cumulative_proportion_accepted
FROM daily
ORDER BY days_to_close;

-- accepted_with_any_related_issue: 1483, 
-- accepted_with_any_open_related_issue: 14,
-- rejected_with_any_related_issue: 802,
-- rejected_with_any_open_related_issue: 292

WITH matching_prs AS (
  SELECT
    pr.id,
    CASE WHEN pr.merged_at IS NOT NULL THEN 1 ELSE 0 END AS is_accepted
  FROM pull_request pr
  JOIN repository repo
    ON repo.id = pr.repo_id
  WHERE pr.state = 'closed'
    AND repo.license IN ('MIT', 'Apache-2.0')
    AND (
      EXISTS (
        SELECT 1
        FROM pr_reviews r
        WHERE r.pr_id = pr.id
          AND r.user <> pr.user
          AND CAST(r.submitted_at AS TIMESTAMP) < CAST(pr.closed_at AS TIMESTAMP)
      )
      OR
      EXISTS (
        SELECT 1
        FROM pr_comments c
        WHERE c.pr_id = pr.id
          AND c.user <> pr.user
          AND CAST(c.created_at AS TIMESTAMP) < CAST(pr.closed_at AS TIMESTAMP)
      )
    )
),
pr_issue_flags AS (
  SELECT
    mp.id AS pr_id,
    mp.is_accepted,

    -- flag: at least one related issue exists
    CASE WHEN EXISTS (
      SELECT 1
      FROM related_issue ri
      WHERE ri.pr_id = mp.id
    ) THEN 1 ELSE 0 END AS has_related_issue,

    -- flag: at least one related issue is still open
    CASE WHEN EXISTS (
      SELECT 1
      FROM related_issue ri
      JOIN issue i
        ON i.id = ri.issue_id
      WHERE ri.pr_id = mp.id
        AND i.state = 'open'
    ) THEN 1 ELSE 0 END AS has_open_related_issue
  FROM matching_prs mp
)
SELECT
  -- Accepted PRs
  SUM(CASE WHEN is_accepted = 1 AND has_related_issue = 1 THEN 1 ELSE 0 END) AS accepted_with_any_related_issue,
  SUM(CASE WHEN is_accepted = 1 AND has_open_related_issue = 1 THEN 1 ELSE 0 END) AS accepted_with_any_open_related_issue,

  -- Rejected PRs (closed but not merged)
  SUM(CASE WHEN is_accepted = 0 AND has_related_issue = 1 THEN 1 ELSE 0 END) AS rejected_with_any_related_issue,
  SUM(CASE WHEN is_accepted = 0 AND has_open_related_issue = 1 THEN 1 ELSE 0 END) AS rejected_with_any_open_related_issue
FROM pr_issue_flags;

-- 

-- pr_status, prs_with_reviews, mean_reviews, median_reviews, stddev_reviews
-- accepted, 3,883, 3.793716, 2, 5.034718
-- rejected, 864, 3.489583, 2, 4.707646

WITH matching_prs AS (
  SELECT
    pr.id,
    CASE WHEN pr.merged_at IS NOT NULL THEN 'accepted' ELSE 'rejected' END AS pr_status
  FROM pull_request pr
  JOIN repository repo
    ON repo.id = pr.repo_id
  WHERE pr.state = 'closed'
    AND repo.license IN ('MIT', 'Apache-2.0')
    AND (
      EXISTS (
        SELECT 1
        FROM pr_reviews r
        WHERE r.pr_id = pr.id
          AND r.user <> pr.user
          AND CAST(r.submitted_at AS TIMESTAMP) < CAST(pr.closed_at AS TIMESTAMP)
      )
      OR
      EXISTS (
        SELECT 1
        FROM pr_comments c
        WHERE c.pr_id = pr.id
          AND c.user <> pr.user
          AND CAST(c.created_at AS TIMESTAMP) < CAST(pr.closed_at AS TIMESTAMP)
      )
    )
),
reviews_per_pr AS (
  SELECT
    mp.id,
    mp.pr_status,
    COUNT(r.id) AS review_count
  FROM matching_prs mp
  LEFT JOIN pr_reviews r
    ON r.pr_id = mp.id
  GROUP BY mp.id, mp.pr_status
),
prs_with_reviews AS (
  SELECT *
  FROM reviews_per_pr
  WHERE review_count >= 1
)
SELECT
  pr_status,
  COUNT(*) AS prs_with_reviews,
  AVG(review_count) AS mean_reviews,
  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY review_count) AS median_reviews,
  STDDEV_POP(review_count) AS stddev_reviews
FROM prs_with_reviews
GROUP BY pr_status
ORDER BY pr_status;

-- Count the review_count, accepted_prs, rejected_prs, total_prs, proportion_accepted, cumulative_proportion_accepted

WITH matching_prs AS (
  SELECT
    pr.id,
    CASE WHEN pr.merged_at IS NOT NULL THEN 1 ELSE 0 END AS is_accepted
  FROM pull_request pr
  JOIN repository repo
    ON repo.id = pr.repo_id
  WHERE pr.state = 'closed'
    AND repo.license IN ('MIT', 'Apache-2.0')
    AND (
      EXISTS (
        SELECT 1
        FROM pr_reviews r
        WHERE r.pr_id = pr.id
          AND r.user <> pr.user
          AND CAST(r.submitted_at AS TIMESTAMP) < CAST(pr.closed_at AS TIMESTAMP)
      )
      OR
      EXISTS (
        SELECT 1
        FROM pr_comments c
        WHERE c.pr_id = pr.id
          AND c.user <> pr.user
          AND CAST(c.created_at AS TIMESTAMP) < CAST(pr.closed_at AS TIMESTAMP)
      )
    )
),
reviews_per_pr AS (
  SELECT
    mp.id,
    mp.is_accepted,
    COUNT(r.id) AS review_count
  FROM matching_prs mp
  LEFT JOIN pr_reviews r
    ON r.pr_id = mp.id
  GROUP BY mp.id, mp.is_accepted
),
by_review_count AS (
  SELECT
    review_count,
    SUM(is_accepted) AS accepted_prs,
    SUM(1 - is_accepted) AS rejected_prs,
    COUNT(*) AS total_prs
  FROM reviews_per_pr
  WHERE review_count >= 1        -- 👈 exclude PRs without reviews
  GROUP BY review_count
)
SELECT
  review_count,
  accepted_prs,
  rejected_prs,
  total_prs,
  accepted_prs * 1.0 / NULLIF(total_prs, 0) AS proportion_accepted,
  SUM(accepted_prs) OVER (ORDER BY review_count) * 1.0
    / NULLIF(SUM(total_prs) OVER (ORDER BY review_count), 0) AS cumulative_proportion_accepted
FROM by_review_count
ORDER BY review_count;

-- pr_status, prs_with_comments, mean_comments, median_comments, stddev_comments
-- accepted, 4,037, 3.494922, 3, 3.419376
-- rejected, 2,091, 3.411765, 3, 2.978198

WITH matching_prs AS (
  SELECT
    pr.id,
    CASE WHEN pr.merged_at IS NOT NULL THEN 'accepted' ELSE 'rejected' END AS pr_status
  FROM pull_request pr
  JOIN repository repo
    ON repo.id = pr.repo_id
  WHERE pr.state = 'closed'
    AND repo.license IN ('MIT', 'Apache-2.0')
    AND (
      EXISTS (
        SELECT 1
        FROM pr_reviews r
        WHERE r.pr_id = pr.id
          AND r.user <> pr.user
          AND CAST(r.submitted_at AS TIMESTAMP) < CAST(pr.closed_at AS TIMESTAMP)
      )
      OR
      EXISTS (
        SELECT 1
        FROM pr_comments c
        WHERE c.pr_id = pr.id
          AND c.user <> pr.user
          AND CAST(c.created_at AS TIMESTAMP) < CAST(pr.closed_at AS TIMESTAMP)
      )
    )
),
comments_per_pr AS (
  SELECT
    mp.id,
    mp.pr_status,
    COUNT(c.id) AS comment_count
  FROM matching_prs mp
  LEFT JOIN pr_comments c
    ON c.pr_id = mp.id
  GROUP BY mp.id, mp.pr_status
),
prs_with_comments AS (
  SELECT *
  FROM comments_per_pr
  WHERE comment_count >= 1
)
SELECT
  pr_status,
  COUNT(*) AS prs_with_comments,
  AVG(comment_count) AS mean_comments,
  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY comment_count) AS median_comments,
  STDDEV_POP(comment_count) AS stddev_comments
FROM prs_with_comments
GROUP BY pr_status
ORDER BY pr_status;

-- Count comment_count, accepted_prs, rejected_prs, total_prs, accepted_prs, proportion_accepted, cumulative_proportion_accepted

WITH matching_prs AS (
  SELECT
    pr.id,
    CASE WHEN pr.merged_at IS NOT NULL THEN 1 ELSE 0 END AS is_accepted
  FROM pull_request pr
  JOIN repository repo
    ON repo.id = pr.repo_id
  WHERE pr.state = 'closed'
    AND repo.license IN ('MIT', 'Apache-2.0')
    AND (
      EXISTS (
        SELECT 1
        FROM pr_reviews r
        WHERE r.pr_id = pr.id
          AND r.user <> pr.user
          AND CAST(r.submitted_at AS TIMESTAMP) < CAST(pr.closed_at AS TIMESTAMP)
      )
      OR
      EXISTS (
        SELECT 1
        FROM pr_comments c
        WHERE c.pr_id = pr.id
          AND c.user <> pr.user
          AND CAST(c.created_at AS TIMESTAMP) < CAST(pr.closed_at AS TIMESTAMP)
      )
    )
),
comments_per_pr AS (
  SELECT
    mp.id,
    mp.is_accepted,
    COUNT(c.id) AS comment_count
  FROM matching_prs mp
  LEFT JOIN pr_comments c
    ON c.pr_id = mp.id
  GROUP BY mp.id, mp.is_accepted
),
by_comment_count AS (
  SELECT
    comment_count,
    SUM(is_accepted) AS accepted_prs,
    SUM(1 - is_accepted) AS rejected_prs,
    COUNT(*) AS total_prs
  FROM comments_per_pr
  WHERE comment_count >= 1              -- 👈 exclude PRs without comments
  GROUP BY comment_count
)
SELECT
  comment_count,
  accepted_prs,
  rejected_prs,
  total_prs,
  accepted_prs * 1.0 / NULLIF(total_prs, 0) AS proportion_accepted,
  SUM(accepted_prs) OVER (ORDER BY comment_count) * 1.0
    / NULLIF(SUM(total_prs) OVER (ORDER BY comment_count), 0) AS cumulative_proportion_accepted
FROM by_comment_count
ORDER BY comment_count;

-- pr_status, prs, mean_commits, median_commits, stddev_commits
-- accepted, 4,963, 4.656659, 3, 5.221643
-- rejected, 2,193, 4.362973, 3, 5.062089

WITH matching_prs AS (
  SELECT
    pr.id,
    CASE WHEN pr.merged_at IS NOT NULL THEN 'accepted' ELSE 'rejected' END AS pr_status
  FROM pull_request pr
  JOIN repository repo
    ON repo.id = pr.repo_id
  WHERE pr.state = 'closed'
    AND repo.license IN ('MIT', 'Apache-2.0')
    AND (
      EXISTS (
        SELECT 1
        FROM pr_reviews r
        WHERE r.pr_id = pr.id
          AND r.user <> pr.user
          AND CAST(r.submitted_at AS TIMESTAMP) < CAST(pr.closed_at AS TIMESTAMP)
      )
      OR
      EXISTS (
        SELECT 1
        FROM pr_comments c
        WHERE c.pr_id = pr.id
          AND c.user <> pr.user
          AND CAST(c.created_at AS TIMESTAMP) < CAST(pr.closed_at AS TIMESTAMP)
      )
    )
),
commits_per_pr AS (
  SELECT
    mp.id,
    mp.pr_status,
    COUNT(pc.sha) AS commit_count
  FROM matching_prs mp
  LEFT JOIN pr_commits pc
    ON pc.pr_id = mp.id
  GROUP BY mp.id, mp.pr_status
)
SELECT
  pr_status,
  COUNT(*) AS prs,
  AVG(commit_count) AS mean_commits,
  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY commit_count) AS median_commits,
  STDDEV_POP(commit_count) AS stddev_commits
FROM commits_per_pr
GROUP BY pr_status
ORDER BY pr_status;

-- Count commit_count, accepted_prs, rejected_prs, total_prs, proportion_accepted, cumulative_proportion_accepted

WITH matching_prs AS (
  SELECT
    pr.id,
    CASE WHEN pr.merged_at IS NOT NULL THEN 1 ELSE 0 END AS is_accepted
  FROM pull_request pr
  JOIN repository repo
    ON repo.id = pr.repo_id
  WHERE pr.state = 'closed'
    AND repo.license IN ('MIT', 'Apache-2.0')
    AND (
      EXISTS (
        SELECT 1
        FROM pr_reviews r
        WHERE r.pr_id = pr.id
          AND r.user <> pr.user
          AND CAST(r.submitted_at AS TIMESTAMP) < CAST(pr.closed_at AS TIMESTAMP)
      )
      OR
      EXISTS (
        SELECT 1
        FROM pr_comments c
        WHERE c.pr_id = pr.id
          AND c.user <> pr.user
          AND CAST(c.created_at AS TIMESTAMP) < CAST(pr.closed_at AS TIMESTAMP)
      )
    )
),
commits_per_pr AS (
  SELECT
    mp.id,
    mp.is_accepted,
    COUNT(pc.sha) AS commit_count
  FROM matching_prs mp
  LEFT JOIN pr_commits pc
    ON pc.pr_id = mp.id
  GROUP BY mp.id, mp.is_accepted
),
by_commit_count AS (
  SELECT
    commit_count,
    SUM(is_accepted) AS accepted_prs,
    SUM(1 - is_accepted) AS rejected_prs,
    COUNT(*) AS total_prs
  FROM commits_per_pr
  WHERE commit_count >= 1          -- exclude PRs with 0 commits
  GROUP BY commit_count
)
SELECT
  commit_count,
  accepted_prs,
  rejected_prs,
  total_prs,
  accepted_prs * 1.0 / NULLIF(total_prs, 0) AS proportion_accepted,
  SUM(accepted_prs) OVER (ORDER BY commit_count) * 1.0
    / NULLIF(SUM(total_prs) OVER (ORDER BY commit_count), 0) AS cumulative_proportion_accepted
FROM by_commit_count
ORDER BY commit_count;

-- pr_status, prs, mean_files_changed, median_files_changed, stddev_files_changed
-- accepted, 4,963, 24.998187, 4, 70.829857
-- rejected, 2,193, 23.403557, 4, 70.668768

WITH matching_prs AS (
  SELECT
    pr.id,
    CASE WHEN pr.merged_at IS NOT NULL THEN 'accepted' ELSE 'rejected' END AS pr_status
  FROM pull_request pr
  JOIN repository repo
    ON repo.id = pr.repo_id
  WHERE pr.state = 'closed'
    AND repo.license IN ('MIT', 'Apache-2.0')
    AND (
      EXISTS (
        SELECT 1
        FROM pr_reviews r
        WHERE r.pr_id = pr.id
          AND r.user <> pr.user
          AND CAST(r.submitted_at AS TIMESTAMP) < CAST(pr.closed_at AS TIMESTAMP)
      )
      OR
      EXISTS (
        SELECT 1
        FROM pr_comments c
        WHERE c.pr_id = pr.id
          AND c.user <> pr.user
          AND CAST(c.created_at AS TIMESTAMP) < CAST(pr.closed_at AS TIMESTAMP)
      )
    )
),
files_per_pr AS (
  SELECT
    mp.id,
    mp.pr_status,
    COUNT(DISTINCT pcd.filename) AS file_count
  FROM matching_prs mp
  LEFT JOIN pr_commit_details pcd
    ON pcd.pr_id = mp.id
  GROUP BY mp.id, mp.pr_status
)
SELECT
  pr_status,
  COUNT(*) AS prs,
  AVG(file_count) AS mean_files_changed,
  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY file_count) AS median_files_changed,
  STDDEV_POP(file_count) AS stddev_files_changed
FROM files_per_pr
GROUP BY pr_status
ORDER BY pr_status;

-- Count changed_files_count, accepted_prs, rejected_prs, total_prs, proportion_accepted, cumulative_proportion_accepted

WITH matching_prs AS (
  SELECT
    pr.id,
    CASE WHEN pr.merged_at IS NOT NULL THEN 1 ELSE 0 END AS is_accepted
  FROM pull_request pr
  JOIN repository repo
    ON repo.id = pr.repo_id
  WHERE pr.state = 'closed'
    AND repo.license IN ('MIT', 'Apache-2.0')
    AND (
      EXISTS (
        SELECT 1
        FROM pr_reviews r
        WHERE r.pr_id = pr.id
          AND r.user <> pr.user
          AND CAST(r.submitted_at AS TIMESTAMP) < CAST(pr.closed_at AS TIMESTAMP)
      )
      OR
      EXISTS (
        SELECT 1
        FROM pr_comments c
        WHERE c.pr_id = pr.id
          AND c.user <> pr.user
          AND CAST(c.created_at AS TIMESTAMP) < CAST(pr.closed_at AS TIMESTAMP)
      )
    )
),
files_per_pr AS (
  SELECT
    mp.id,
    mp.is_accepted,
    COUNT(DISTINCT pcd.filename) AS changed_files_count
  FROM matching_prs mp
  LEFT JOIN pr_commit_details pcd
    ON pcd.pr_id = mp.id
  GROUP BY mp.id, mp.is_accepted
),
by_file_count AS (
  SELECT
    changed_files_count,
    SUM(is_accepted) AS accepted_prs,
    SUM(1 - is_accepted) AS rejected_prs,
    COUNT(*) AS total_prs
  FROM files_per_pr
  WHERE changed_files_count >= 1        -- exclude PRs with 0 changed files
  GROUP BY changed_files_count
)
SELECT
  changed_files_count,
  accepted_prs,
  rejected_prs,
  total_prs,
  accepted_prs * 1.0 / NULLIF(total_prs, 0) AS proportion_accepted,
  SUM(accepted_prs) OVER (ORDER BY changed_files_count) * 1.0
    / NULLIF(SUM(total_prs) OVER (ORDER BY changed_files_count), 0)
      AS cumulative_proportion_accepted
FROM by_file_count
ORDER BY changed_files_count;