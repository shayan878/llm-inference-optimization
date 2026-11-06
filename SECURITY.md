# Security policy

Do not publish credentials, access tokens, private model files, or infrastructure secrets in issues or pull requests. Report suspected vulnerabilities privately to the repository owner through GitHub's private vulnerability reporting feature when available.

The example deployment script avoids broad container privileges by default. Review all host paths, network exposure, model sources, and Docker image tags before use.
