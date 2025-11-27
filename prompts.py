"""
Prompts for fake review detection system.
Contains three experimental conditions: Zero-Shot, Few-Shot, and Few-Shot with Chain-of-Thought (CoT).
"""

# ==============================================================================
# CONDITION A: ZERO-SHOT CLASSIFICATION
# ==============================================================================

ZERO_SHOT_PROMPT = """You are an expert review authenticity analyst specializing in detecting fraudulent and fake reviews. Your task is to analyze restaurant reviews and determine whether they are genuine or fake based on linguistic patterns, reviewer behavior, and contextual signals.

## Review Information to Analyze:

**Review Content:** {reviewContent}

**Review Metadata:**
- Star Rating Given: {rating}/5
- Review Usefulness Count: {reviewUsefulCount} users found this helpful
- Review Length: {ReviewLength} characters


**Reviewer Profile:**
- Total Reviews Written: {reviewCount}
- Yelp Member Since: {yelpJoinDate}
- Friend Count: {friendCount}
- Total Useful Votes Received: {usefulCount}
- Cool Votes: {coolCount}
- Funny Votes: {funnyCount}
- Compliment Count: {complimentCount}

**Restaurant Context:**
- Restaurant Overall Rating: {restaurantRating}/5

## Detection Guidelines:

**Indicators of FAKE Reviews:**
1. Overly generic language lacking specific details about food, service, or ambiance
2. Extreme sentiment (all positive or all negative) without nuanced observations
3. Suspicious reviewer profile: new account, very few reviews, no social engagement
4. Rating significantly deviates from restaurant's overall rating without justified reasoning
5. Repetitive phrases, unnatural language patterns, or promotional tone
6. Lack of personal experience markers (e.g., "I ordered", "my waiter", specific menu items)
7. Short length with extreme ratings (either 1 or 5 stars)
8. Focus on competitors or irrelevant information rather than actual experience

**Indicators of REAL Reviews:**
1. Specific details about dishes, prices, service interactions, or atmosphere
2. Balanced perspective mentioning both positives and negatives
3. Established reviewer with history of engagement (reviews, friends, votes)
4. Natural, conversational language with personal anecdotes
5. Contextual information (occasion, time of visit, who they went with)
6. Appropriate length and detail matching the rating given
7. Constructive criticism or genuine enthusiasm with supporting examples

## Output Requirements:

You must respond with ONLY a valid JSON object in the following format:

{{
    "flag": "fake" | "real",
    "reasoning": "Concise explanation citing specific evidence from the review content and metadata that supports your classification"
}}

**Important:**
- Base your analysis ONLY on the information provided above
- Do NOT make assumptions about information not given
- The "reasoning" field should be 2-4 sentences explaining the key factors that led to your decision
- Reference specific textual cues, patterns, or metadata values in your reasoning
- Be objective and analytical in your assessment

Analyze the review now and provide your response:"""


# ==============================================================================
# CONDITION B: FEW-SHOT CLASSIFICATION
# ==============================================================================

FEW_SHOT_PROMPT = """You are an expert review authenticity analyst specializing in detecting fraudulent and fake reviews. Your task is to analyze restaurant reviews and determine whether they are genuine or fake based on linguistic patterns, reviewer behavior, and contextual signals.

## Training Examples:

### Example 1 - REAL Review:
**Review Content:** "I went here with my family last Sunday for brunch. We ordered the eggs benedict, buttermilk pancakes, and the breakfast burrito. The hollandaise sauce was perfectly lemony and the english muffin was crispy. Service was a bit slow - took about 15 minutes to get our coffee - but our server Sarah was very apologetic and friendly. The pancakes were huge, my daughter couldn't finish hers. Prices are reasonable, about $45 for three of us. Would definitely come back, maybe on a weekday when it's less crowded."
**Metadata:** Rating: 4/5 | Reviewer Total Reviews: 47 | Member Since: March 2018 | Friend Count: 12 | Useful Votes: 89 | Restaurant Rating: 4.0/5
**Classification:** {{"flag": "real", "reasoning": "Review contains specific menu items, detailed sensory descriptions, mentions staff by name, includes minor criticism, provides context about visit timing and companions, and reviewer has established history."}}

### Example 2 - REAL Review:
**Review Content:** "Honestly disappointed. I'd heard great things but the pasta was overcooked and mushy. The carbonara had way too much cream - it's supposed to be just eggs and cheese. My boyfriend's pizza was better, thin crust with good char, but $18 for a margherita seems steep. The ambiance is nice though, dim lighting and exposed brick. Probably won't return but I can see why some people like it."
**Metadata:** Rating: 2/5 | Reviewer Total Reviews: 134 | Member Since: June 2016 | Friend Count: 28 | Useful Votes: 412 | Restaurant Rating: 4.5/5
**Classification:** {{"flag": "real", "reasoning": "Review shows culinary knowledge, provides balanced perspective with both criticism and praise, includes price point, mentions companion, demonstrates understanding of dish preparation, and reviewer is established with significant engagement history."}}

### Example 3 - FAKE Review:
**Review Content:** "Amazing restaurant best food ever everything was perfect the service was outstanding I highly recommend this place to everyone you will not be disappointed trust me this is the best restaurant in the city go here now"
**Metadata:** Rating: 5/5 | Reviewer Total Reviews: 2 | Member Since: January 2023 | Friend Count: 0 | Useful Votes: 0 | Restaurant Rating: 3.0/5
**Classification:** {{"flag": "fake", "reasoning": "Review uses excessive generic praise without specific details, lacks punctuation suggesting automated generation, no mention of actual dishes or experiences, extreme 5-star rating conflicts with restaurant's 3.0 average, and reviewer profile shows minimal engagement with only 2 total reviews and no social connections."}}

### Example 4 - FAKE Review:
**Review Content:** "Terrible experience worst place ever the food was disgusting and the staff was rude everything was bad don't waste your money go anywhere else but here I will never come back"
**Metadata:** Rating: 1/5 | Reviewer Total Reviews: 1 | Member Since: May 2023 | Friend Count: 0 | Useful Votes: 0 | Restaurant Rating: 4.5/5
**Classification:** {{"flag": "fake", "reasoning": "Review contains only extreme negative generalities without specific incidents or details, lacks proper punctuation, provides no concrete examples of rudeness or food quality issues, 1-star rating drastically conflicts with restaurant's 4.5 average, and reviewer is newly created account with single review and no engagement history."}}

## Review Information to Analyze:

**Review Content:** {reviewContent}

**Review Metadata:**
- Star Rating Given: {rating}/5
- Review Usefulness Count: {reviewUsefulCount} users found this helpful
- Review Length: {ReviewLength} characters

**Reviewer Profile:**
- Total Reviews Written: {reviewCount}
- Yelp Member Since: {yelpJoinDate}
- Friend Count: {friendCount}
- Total Useful Votes Received: {usefulCount}
- Cool Votes: {coolCount}
- Funny Votes: {funnyCount}
- Compliment Count: {complimentCount}

**Restaurant Context:**
- Restaurant Overall Rating: {restaurantRating}/5

## Output Requirements:

You must respond with ONLY a valid JSON object in the following format:

{{
    "flag": "fake" | "real",
    "reasoning": "Concise explanation citing specific evidence from the review content and metadata that supports your classification"
}}

**Important:**
- Base your analysis ONLY on the information provided
- The "reasoning" field should be 2-4 sentences explaining the key factors
- Reference specific textual cues, patterns, or metadata values in your reasoning

Analyze the review now and provide your response:"""


# ==============================================================================
# CONDITION C: FEW-SHOT CLASSIFICATION WITH CHAIN-OF-THOUGHT (CoT)
# ==============================================================================

FEW_SHOT_COT_PROMPT = """You are an expert review authenticity analyst specializing in detecting fraudulent and fake reviews. Your task is to analyze restaurant reviews and determine whether they are genuine or fake based on linguistic patterns, reviewer behavior, and contextual signals.

You must use a step-by-step reasoning process before making your final classification.

## Training Examples with Chain-of-Thought Reasoning:

### Example 1 - REAL Review:
**Review Content:** "I went here with my family last Sunday for brunch. We ordered the eggs benedict, buttermilk pancakes, and the breakfast burrito. The hollandaise sauce was perfectly lemony and the english muffin was crispy. Service was a bit slow - took about 15 minutes to get our coffee - but our server Sarah was very apologetic and friendly. The pancakes were huge, my daughter couldn't finish hers. Prices are reasonable, about $45 for three of us. Would definitely come back, maybe on a weekday when it's less crowded."
**Metadata:** Rating: 4/5 | Reviewer Total Reviews: 47 | Member Since: March 2018 | Friend Count: 12 | Useful Votes: 89 | Restaurant Rating: 4.0/5

**Step-by-Step Analysis:**
1. **Content Specificity:** Review mentions three specific menu items (eggs benedict, buttermilk pancakes, breakfast burrito) with detailed sensory descriptions (lemony hollandaise, crispy muffin). This level of detail suggests genuine experience.
2. **Balanced Perspective:** Includes both positive elements (food quality, friendly server) and negative criticism (slow service, 15-minute wait). Real reviewers typically provide nuanced feedback.
3. **Personal Context:** References family visit, daughter's inability to finish meal, plans to return on weekday. These personal markers are difficult for fake reviewers to fabricate authentically.
4. **Reviewer Credibility:** 47 total reviews since 2018, 12 friends, 89 useful votes indicate established, engaged user rather than suspicious new account.
5. **Rating Alignment:** 4/5 rating matches restaurant's 4.0 average and is consistent with mixed positive/negative feedback.

**Classification:** {{"flag": "real", "reasoning": "Review demonstrates genuine experience through specific menu details, balanced critique, personal context markers, established reviewer profile with 5-year history, and rating alignment with feedback content."}}

### Example 2 - FAKE Review:
**Review Content:** "Amazing restaurant best food ever everything was perfect the service was outstanding I highly recommend this place to everyone you will not be disappointed trust me this is the best restaurant in the city go here now"
**Metadata:** Rating: 5/5 | Reviewer Total Reviews: 2 | Member Since: January 2023 | Friend Count: 0 | Useful Votes: 0 | Restaurant Rating: 3.0/5

**Step-by-Step Analysis:**
1. **Generic Language:** Uses vague superlatives ("best food ever," "everything was perfect," "outstanding") without any specific details about dishes, ambiance, or service interactions.
2. **Linguistic Patterns:** Lacks punctuation and reads as a continuous stream, suggesting possible automated generation or non-native speaker following a template.
3. **Suspicious Enthusiasm:** Extreme promotional tone with urgent language ("go here now," "trust me") is atypical of genuine reviews and common in paid/incentivized reviews.
4. **Reviewer Profile Red Flags:** Only 2 total reviews, brand new account (2023), zero friends, zero useful votes indicates no established presence or credibility.
5. **Rating Discrepancy:** 5-star rating conflicts sharply with restaurant's 3.0 overall rating without justified explanation for the extreme positive deviation.
6. **No Experiential Details:** Complete absence of what was ordered, when visited, who accompanied, price points, or any verifiable experience markers.

**Classification:** {{"flag": "fake", "reasoning": "Review exhibits multiple deception indicators including generic praise without specifics, suspicious promotional tone, lack of punctuation suggesting automation, severely misaligned rating versus restaurant average, and brand new account with minimal activity suggesting possible astroturfing."}}

### Example 3 - REAL Review:
**Review Content:** "Honestly disappointed. I'd heard great things but the pasta was overcooked and mushy. The carbonara had way too much cream - it's supposed to be just eggs and cheese. My boyfriend's pizza was better, thin crust with good char, but $18 for a margherita seems steep. The ambiance is nice though, dim lighting and exposed brick. Probably won't return but I can see why some people like it."
**Metadata:** Rating: 2/5 | Reviewer Total Reviews: 134 | Member Since: June 2016 | Friend Count: 28 | Useful Votes: 412 | Restaurant Rating: 4.5/5

**Step-by-Step Analysis:**
1. **Culinary Knowledge:** Demonstrates understanding of proper carbonara preparation (eggs and cheese, not cream), suggesting genuine food knowledge rather than scripted feedback.
2. **Specific Criticisms:** Identifies exact issues (overcooked pasta, incorrect sauce, high price) with concrete examples rather than generic complaints.
3. **Nuanced Perspective:** Despite low rating, acknowledges positive aspects (boyfriend's pizza quality, nice ambiance) and even empathizes with others who might enjoy it.
4. **Price Transparency:** Mentions specific price point ($18 for margherita) allowing readers to make informed decisions.
5. **High Credibility Profile:** 134 reviews since 2016, 28 friends, 412 useful votes demonstrates long-term engaged user with community trust.
6. **Rating Deviation Justified:** While 2-star rating is below restaurant's 4.5 average, the detailed negative experiences provide clear justification.

**Classification:** {{"flag": "real", "reasoning": "Review shows authentic experience through culinary knowledge, specific verifiable details, balanced perspective despite disappointment, transparent pricing, and highly credible reviewer profile with 7+ years of platform engagement."}}

### Example 4 - FAKE Review:
**Review Content:** "Terrible experience worst place ever the food was disgusting and the staff was rude everything was bad don't waste your money go anywhere else but here I will never come back"
**Metadata:** Rating: 1/5 | Reviewer Total Reviews: 1 | Member Since: May 2023 | Friend Count: 0 | Useful Votes: 0 | Restaurant Rating: 4.5/5

**Step-by-Step Analysis:**
1. **Lack of Specificity:** Uses only generic negative terms ("terrible," "worst," "disgusting," "bad") without describing what made food disgusting or how staff was rude.
2. **Linguistic Red Flags:** No punctuation, run-on sentence structure suggests automated generation or template following.
3. **Extreme Negativity:** Absolute language ("worst place ever," "everything was bad") without a single specific incident or detail is characteristic of malicious fake reviews.
4. **No Actionable Information:** Fails to mention what was ordered, when visited, specific staff interactions, or any verifiable details that management could address.
5. **Suspicious Profile:** Brand new account with only 1 review, no friends, no useful votes - classic pattern of account created specifically to leave damaging review.
6. **Severe Rating Conflict:** 1-star rating drastically conflicts with restaurant's strong 4.5 average without justified explanation, suggesting possible competitor sabotage or grudge posting.

**Classification:** {{"flag": "fake", "reasoning": "Review demonstrates clear deception markers including zero specific details despite extreme claims, lack of punctuation indicating automation, newly created single-purpose account with no engagement history, and unjustified 1-star rating conflicting with established 4.5 restaurant average."}}

## Review Information to Analyze:

**Review Content:** {reviewContent}

**Review Metadata:**
- Star Rating Given: {rating}/5
- Review Usefulness Count: {reviewUsefulCount} users found this helpful
- Review Length: {ReviewLength} characters


**Reviewer Profile:**
- Total Reviews Written: {reviewCount}
- Yelp Member Since: {yelpJoinDate}
- Friend Count: {friendCount}
- Total Useful Votes Received: {usefulCount}
- Cool Votes: {coolCount}
- Funny Votes: {funnyCount}
- Compliment Count: {complimentCount}

**Restaurant Context:**
- Restaurant Overall Rating: {restaurantRating}/5

## Analysis Instructions:

You MUST follow this step-by-step reasoning process:

1. **Content Specificity Analysis:** Evaluate whether the review contains specific, verifiable details (menu items, prices, staff names, dates) or generic statements.

2. **Linguistic Pattern Assessment:** Examine language quality, punctuation, tone, and whether it reads naturally or appears automated/templated.

3. **Sentiment Balance Evaluation:** Determine if the review provides nuanced perspective with both positives and negatives, or is extremely one-sided.

4. **Reviewer Credibility Check:** Assess the reviewer's profile metrics (account age, review count, social engagement) for signs of legitimate or suspicious activity.

5. **Rating Alignment Analysis:** Compare the given rating with the restaurant's overall rating and the review content to identify inconsistencies.

6. **Experiential Markers Verification:** Look for personal context, visit details, companions, occasions, or other markers that indicate genuine experience.

## Output Requirements:

You must respond with ONLY a valid JSON object in the following format:

{{
    "flag": "fake" | "real",
    "reasoning": "Synthesize your step-by-step analysis into 3-5 sentences explaining the key factors that led to your classification, referencing specific evidence from both content and metadata"
}}

**Important:**
- You must mentally work through all 6 analysis steps before providing your final classification
- Base your analysis ONLY on the information provided
- Reference specific evidence from your reasoning process in the final reasoning field

Analyze the review now and provide your response:"""


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def format_zero_shot_prompt(row):
    """
    Format the zero-shot classification prompt with actual data from a dataframe row.
    
    Condition A: Tests the model's intrinsic capability without any examples.
    
    Args:
        row: A pandas Series or dict containing review data with required columns
        
    Returns:
        str: Formatted zero-shot prompt ready for LLM inference
    """
    return ZERO_SHOT_PROMPT.format(
        reviewContent=row.get('reviewContent', 'N/A'),
        rating=row.get('rating', 'N/A'),
        reviewUsefulCount=row.get('reviewUsefulCount', 0),
        ReviewLength=row.get('ReviewLength', 0),
        date=row.get('date', 'N/A'),
        reviewCount=row.get('reviewCount', 0),
        yelpJoinDate=row.get('yelpJoinDate', 'N/A'),
        friendCount=row.get('friendCount', 0),
        usefulCount=row.get('usefulCount', 0),
        coolCount=row.get('coolCount', 0),
        funnyCount=row.get('funnyCount', 0),
        complimentCount=row.get('complimentCount', 0),
        restaurantRating=row.get('restaurantRating', 'N/A')
    )


def format_few_shot_prompt(row):
    """
    Format the few-shot classification prompt with actual data from a dataframe row.
    
    Condition B: Tests the model's in-context learning with labeled examples.
    
    Args:
        row: A pandas Series or dict containing review data with required columns
        
    Returns:
        str: Formatted few-shot prompt ready for LLM inference
    """
    return FEW_SHOT_PROMPT.format(
        reviewContent=row.get('reviewContent', 'N/A'),
        rating=row.get('rating', 'N/A'),
        reviewUsefulCount=row.get('reviewUsefulCount', 0),
        ReviewLength=row.get('ReviewLength', 0),
        date=row.get('date', 'N/A'),
        reviewCount=row.get('reviewCount', 0),
        yelpJoinDate=row.get('yelpJoinDate', 'N/A'),
        friendCount=row.get('friendCount', 0),
        usefulCount=row.get('usefulCount', 0),
        coolCount=row.get('coolCount', 0),
        funnyCount=row.get('funnyCount', 0),
        complimentCount=row.get('complimentCount', 0),
        restaurantRating=row.get('restaurantRating', 'N/A')
    )


def format_few_shot_cot_prompt(row):
    """
    Format the few-shot Chain-of-Thought classification prompt with actual data.
    
    Condition C: Tests structured reasoning by requiring step-by-step analysis.
    
    Args:
        row: A pandas Series or dict containing review data with required columns
        
    Returns:
        str: Formatted few-shot CoT prompt ready for LLM inference
    """
    return FEW_SHOT_COT_PROMPT.format(
        reviewContent=row.get('reviewContent', 'N/A'),
        rating=row.get('rating', 'N/A'),
        reviewUsefulCount=row.get('reviewUsefulCount', 0),
        ReviewLength=row.get('ReviewLength', 0),
        date=row.get('date', 'N/A'),
        reviewCount=row.get('reviewCount', 0),
        yelpJoinDate=row.get('yelpJoinDate', 'N/A'),
        friendCount=row.get('friendCount', 0),
        usefulCount=row.get('usefulCount', 0),
        coolCount=row.get('coolCount', 0),
        funnyCount=row.get('funnyCount', 0),
        complimentCount=row.get('complimentCount', 0),
        restaurantRating=row.get('restaurantRating', 'N/A')
    )


def get_prompt_formatter(condition='zero_shot'):
    """
    Get the appropriate prompt formatter function based on experimental condition.
    
    Args:
        condition: One of 'zero_shot', 'few_shot', or 'few_shot_cot'
        
    Returns:
        function: The corresponding formatter function
        
    Raises:
        ValueError: If condition is not recognized
    """
    formatters = {
        'zero_shot': format_zero_shot_prompt,
        'few_shot': format_few_shot_prompt,
        'few_shot_cot': format_few_shot_cot_prompt
    }
    
    if condition not in formatters:
        raise ValueError(
            f"Unknown condition '{condition}'. "
            f"Must be one of: {list(formatters.keys())}"
        )
    
    return formatters[condition]


# Backward compatibility alias
format_detection_prompt = format_zero_shot_prompt

