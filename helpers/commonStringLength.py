
# def longest_common_substring(str1, str2):
#     # Create a 2D table to store lengths of longest common suffixes
#     m, n = len(str1), len(str2)
#     dp = [[0] * (n + 1) for _ in range(m + 1)]
    
#     # Variables to store the length of the longest common substring and its position
#     longest = 0
#     end_pos = 0  # End position of the longest substring in str1
    
#     # Fill the dp table
#     for i in range(1, m + 1):
#         for j in range(1, n + 1):
#             if str1[i - 1] == str2[j - 1]:  # Characters match
#                 dp[i][j] = dp[i - 1][j - 1] + 1
#                 if dp[i][j] > longest:
#                     longest = dp[i][j]
#                     end_pos = i
#             else:
#                 dp[i][j] = 0
    
#     # Extract the longest common substring
#     longest_substring = str1[end_pos - longest:end_pos]
#     return len(longest_substring)


def longest_common_substring(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    longest = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                longest = max(longest, dp[i][j])
            else:
                dp[i][j] = 0
    return longest