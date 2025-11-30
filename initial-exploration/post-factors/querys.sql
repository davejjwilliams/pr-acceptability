-- 1. TBL pull_request 
-- 1.1 Average days to closed for accepted PRs
SELECT AVG(date_diff('days', created_at::DATE, closed_at::DATE)) AS days_to_merge
	,STDDEV_POP(date_diff('days', created_at::DATE, closed_at::DATE)) AS stddev_days_to_merge
FROM pull_request
WHERE STATE = 'closed'
	AND merged_at IS NOT NULL;

-- 1.2 Counts of accepted PRs by days to close	
SELECT date_diff('days', created_at::DATE, closed_at::DATE) AS days_to_close
	,COUNT(*) AS num_accepted_prs
FROM pull_request
WHERE STATE = 'closed'
	AND merged_at IS NOT NULL
GROUP BY days_to_close
ORDER BY days_to_close;

-- 2. TBL pr_reviews
-- 2.1 Average reviews for accepted PRs
WITH closed_merged_prs
AS (
	SELECT id
	FROM pull_request
	WHERE STATE = 'closed'
		AND merged_at IS NOT NULL
	)
	,review_counts
AS (
	SELECT pr_id
		,COUNT(*) AS review_count
	FROM pr_reviews
	WHERE pr_id IN (
			SELECT id
			FROM closed_merged_prs
			)
	GROUP BY pr_id
	)
SELECT (
		SELECT COUNT(*)
		FROM review_counts
		) AS num_prs_with_reviews
	,(
		SELECT AVG(review_count)
		FROM review_counts
		) AS avg_reviews_for_prs_with_reviews
	,(
		SELECT STDDEV_POP(review_count)
		FROM review_counts
		) AS stddev_reviews_for_prs_with_reviews;

-- 2.2 Counts of accepted PRs with comments
WITH closed_merged_prs
AS (
	SELECT id
	FROM pull_request
	WHERE STATE = 'closed'
		AND merged_at IS NOT NULL
	)
	,reviews_for_closed_prs
AS (
	SELECT id AS review_id
		,pr_id
	FROM pr_reviews
	WHERE pr_id IN (
			SELECT id
			FROM closed_merged_prs
			)
	)
	,reviews_with_comments
AS (
	SELECT DISTINCT pull_request_review_id AS review_id
	FROM pr_review_comments_v2
	)
	,prs_with_comments
AS (
	SELECT DISTINCT r.pr_id
	FROM reviews_for_closed_prs r
	JOIN reviews_with_comments c ON r.review_id = c.review_id
	)
SELECT COUNT(*) AS closed_prs_with_comments
FROM prs_with_comments;

-- 3. TBL related_issues
-- 3.1 Counts of accepted PRs with related issues
SELECT COUNT(DISTINCT pr_id) AS prs_with_issues
FROM related_issue
WHERE pr_id IN (
		SELECT id
		FROM pull_request
		WHERE STATE = 'closed'
			AND merged_at IS NOT NULL
		);

WITH closed_merged_prs
AS (
	SELECT id
	FROM pull_request
	WHERE STATE = 'closed'
		AND merged_at IS NOT NULL
	)
	,issue_counts
AS (
	SELECT pr_id
		,COUNT(*) AS issue_count
	FROM related_issue
	WHERE pr_id IN (
			SELECT id
			FROM closed_merged_prs
			)
	GROUP BY pr_id
	)
SELECT COUNT(*) AS prs_with_issues
	,AVG(issue_count) AS avg_issues_per_pr_with_issues
FROM issue_counts;

-- 4. TBL pr_commits 
-- 4.1 Average commits for accepted PRs
WITH closed_merged_prs
AS (
	SELECT id
	FROM pull_request
	WHERE STATE = 'closed'
		AND merged_at IS NOT NULL
	)
	,commit_counts
AS (
	SELECT pr_id
		,COUNT(*) AS commit_count
	FROM pr_commits
	WHERE pr_id IN (
			SELECT id
			FROM closed_merged_prs
			)
	GROUP BY pr_id
	)
SELECT COUNT(*) AS num_prs_with_commits
	,AVG(commit_count) AS avg_commits_for_prs_with_commits
	,STDDEV_POP(commit_count) AS stddev_commits_for_prs_with_commits
FROM commit_counts;

-- 4.2 Counts of accepted PRs by number of commits
WITH closed_merged_prs
AS (
	SELECT id
	FROM pull_request
	WHERE STATE = 'closed'
		AND merged_at IS NOT NULL
	)
	,commit_counts
AS (
	SELECT pr_id
		,COUNT(*) AS commit_count
	FROM pr_commits
	WHERE pr_id IN (
			SELECT id
			FROM closed_merged_prs
			)
	GROUP BY pr_id
	)
SELECT commit_count
	,COUNT(*) AS num_prs
FROM commit_counts
GROUP BY commit_count
ORDER BY commit_count;

-- 5. TBL pr_commit_details 
-- 5.1 Averge changes for accepted PRs
WITH accepted_prs
AS (
	SELECT id
	FROM pull_request
	WHERE STATE = 'closed'
		AND merged_at IS NOT NULL
	)
	,changes_per_pr
AS (
	SELECT pr_id
		,SUM(changes) AS total_changes
	FROM pr_commit_details
	WHERE pr_id IN (
			SELECT id
			FROM accepted_prs
			)
	GROUP BY pr_id
	)
SELECT AVG(total_changes) AS avg_changes_per_pr
	,STDDEV_POP(total_changes) AS stddev_changes_per_pr
FROM changes_per_pr;

-- 5.2 Counts of accepted PRs by invertals of changes
WITH closed_merged_prs
AS (
	SELECT id
	FROM pull_request
	WHERE STATE = 'closed'
		AND merged_at IS NOT NULL
	)
	,commit_changes
AS (
	SELECT pr_id
		,SUM(changes) AS total_changes
	FROM pr_commit_details
	GROUP BY pr_id
	)
SELECT AVG(total_changes) AS avg_changes_per_pr
	,STDDEV_POP(total_changes) AS std_changes_per_pr
FROM commit_changes
WHERE pr_id IN (
		SELECT id
		FROM closed_merged_prs
		);

WITH closed_merged_prs
AS (
	SELECT id
	FROM pull_request
	WHERE STATE = 'closed'
		AND merged_at IS NOT NULL
	)
	,commit_changes
AS (
	SELECT pr_id
		,SUM(changes) AS total_changes
	FROM pr_commit_details
	GROUP BY pr_id
	)
SELECT CASE 
		WHEN total_changes <= 1000
			THEN '≤1000'
		WHEN total_changes BETWEEN 1001
				AND 2000
			THEN '1001–2000'
		WHEN total_changes BETWEEN 2001
				AND 3000
			THEN '2001–3000'
		WHEN total_changes BETWEEN 3001
				AND 4000
			THEN '3001–4000'
		WHEN total_changes BETWEEN 4001
				AND 5000
			THEN '4001–5000'
		WHEN total_changes BETWEEN 5001
				AND 6000
			THEN '5001–6000'
		WHEN total_changes BETWEEN 6001
				AND 7000
			THEN '6001–7000'
		WHEN total_changes BETWEEN 7001
				AND 8000
			THEN '7001–8000'
		WHEN total_changes BETWEEN 8001
				AND 9000
			THEN '8001–9000'
		ELSE '≥9000'
		END AS change_bucket
	,COUNT(*) AS num_prs
FROM commit_changes
WHERE pr_id IN (
		SELECT id
		FROM closed_merged_prs
		)
GROUP BY change_bucket
ORDER BY CASE change_bucket
		WHEN '≤1000'
			THEN 1
		WHEN '1001–2000'
			THEN 2
		WHEN '2001–3000'
			THEN 3
		WHEN '3001–4000'
			THEN 4
		WHEN '4001–5000'
			THEN 5
		WHEN '5001–6000'
			THEN 6
		WHEN '6001–7000'
			THEN 7
		WHEN '7001–8000'
			THEN 8
		WHEN '8001–9000'
			THEN 9
		WHEN '≥9000'
			THEN 10
		END;

-- 6 TBL pr_task_type
-- 6.1 Counts of pull request by task type
WITH accepted_prs
AS (
	SELECT id
	FROM pull_request
	WHERE STATE = 'closed'
		AND merged_at IS NOT NULL
	)
SELECT type
	,COUNT(*) AS num_prs
FROM pr_task_type
WHERE id IN (
		SELECT id
		FROM accepted_prs
		)
GROUP BY type
ORDER BY num_prs DESC;
