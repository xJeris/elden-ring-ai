"""
Goal System for Elden Ring AI
Defines what the AI needs to accomplish and tracks progress
"""

class GameGoal:
    """Represents a goal the AI should work toward"""
    
    def __init__(self, name, description, priority=1, reward_value=100, sub_goals=None, required_sub_goals=None):
        """
        Initialize a goal
        
        Args:
            name: short goal name (e.g., "Defeat Godrick")
            description: detailed description
            priority: 1-10 (higher = more important)
            reward_value: reward points for completing this goal
            sub_goals: list of sub-goal dicts with format:
                      [{"name": "Defeat Margit", "reward": 300}, ...]
            required_sub_goals: how many sub-goals must be completed (default: all)
                               Can be int (e.g., 2) or "all"
        """
        self.name = name
        self.description = description
        self.priority = priority
        self.reward_value = reward_value
        self.completed = False
        self.progress = 0  # 0-100%
        self.detection_method = None
        
        # Sub-goals tracking
        self.sub_goals = sub_goals or []
        self.completed_sub_goals = []
        
        # How many sub-goals are required to complete this goal
        if required_sub_goals is None or required_sub_goals == "all":
            self.required_sub_goals = len(self.sub_goals) if self.sub_goals else 1
        else:
            self.required_sub_goals = required_sub_goals
    
    def add_sub_goal(self, name, reward=100):
        """Add a sub-goal to this goal"""
        self.sub_goals.append({
            "name": name,
            "reward": reward,
            "completed": False
        })
    
    def complete_sub_goal(self, sub_goal_name):
        """Mark a sub-goal as complete"""
        for sub_goal in self.sub_goals:
            if sub_goal["name"] == sub_goal_name and not sub_goal["completed"]:
                sub_goal["completed"] = True
                self.completed_sub_goals.append(sub_goal_name)
                
                # Update progress based on completed sub-goals vs required
                completed_count = sum(1 for sg in self.sub_goals if sg["completed"])
                self.progress = int((completed_count / self.required_sub_goals) * 100)
                
                # Check if goal is complete
                if completed_count >= self.required_sub_goals:
                    self.completed = True
                
                return True
        return False
    
    def get_sub_goal_reward(self):
        """Calculate reward from completed sub-goals"""
        reward = 0
        for sub_goal in self.sub_goals:
            if sub_goal["completed"]:
                reward += sub_goal["reward"]
        return reward
    
    def get_required_sub_goals(self):
        """Get list of sub-goals that haven't been completed yet"""
        return [sg for sg in self.sub_goals if not sg["completed"]]
    
    def get_progress_summary(self):
        """Get readable progress summary"""
        completed = sum(1 for sg in self.sub_goals if sg["completed"])
        if self.required_sub_goals == len(self.sub_goals):
            return f"{completed}/{len(self.sub_goals)} sub-goals"
        else:
            return f"{completed}/{self.required_sub_goals} sub-goals required (out of {len(self.sub_goals)} options)"
    
    def __repr__(self):
        status = "✓" if self.completed else "○"
        return f"{status} [{self.priority}] {self.name} ({self.progress}%)"


class GoalSystem:
    """Manages all goals for the AI"""
    
    def __init__(self):
        self.goals = []
        self.main_goal = None
    
    def add_goal(self, goal):
        """Add a goal to the system"""
        self.goals.append(goal)
    
    def set_main_objective(self, goal_name):
        """Set the primary goal the AI should focus on"""
        for goal in self.goals:
            if goal.name == goal_name:
                self.main_goal = goal
                print(f"Main objective set to: {goal_name}")
                return True
        print(f"Goal '{goal_name}' not found!")
        return False
    
    def get_goal_reward(self):
        """Calculate reward based on goal progress"""
        total_reward = 0
        
        # Reward completed goals heavily
        for goal in self.goals:
            if goal.completed:
                total_reward += goal.reward_value
            
            # Also reward completed sub-goals
            total_reward += goal.get_sub_goal_reward()
        
        # Reward progress on main goal
        if self.main_goal:
            progress_reward = (self.main_goal.progress / 100) * (self.main_goal.reward_value * 0.5)
            total_reward += progress_reward
        
        return total_reward
    
    def list_goals(self):
        """Display all goals"""
        print("\n" + "=" * 60)
        print("CURRENT GOALS")
        print("=" * 60)
        
        if not self.goals:
            print("No goals set!")
            return
        
        # Sort by priority
        sorted_goals = sorted(self.goals, key=lambda x: x.priority, reverse=True)
        
        for goal in sorted_goals:
            status = "✓ COMPLETED" if goal.completed else f"IN PROGRESS ({goal.progress}%)"
            print(f"\n[Priority {goal.priority}] {goal.name}")
            print(f"  Status: {status}")
            print(f"  Description: {goal.description}")
            print(f"  Reward: {goal.reward_value} points")
            
            # Show sub-goals if any
            if goal.sub_goals:
                print(f"\n  Sub-Goals: {goal.get_progress_summary()}")
                for sub_goal in goal.sub_goals:
                    sub_status = "✓" if sub_goal["completed"] else "○"
                    print(f"    {sub_status} {sub_goal['name']} (+{sub_goal['reward']} points)")
                
                # Show remaining sub-goals needed
                remaining = goal.get_required_sub_goals()
                completed = sum(1 for sg in goal.sub_goals if sg["completed"])
                still_needed = goal.required_sub_goals - completed
                
                if still_needed > 0:
                    print(f"\n  ⓘ Need {still_needed} more sub-goal(s) to complete this goal")
            
            if goal.progress > 0 and not goal.completed:
                progress_bar = "█" * int(goal.progress / 5) + "░" * (20 - int(goal.progress / 5))
                print(f"  Progress: [{progress_bar}]")
        
        print("\n" + "=" * 60)


def create_base_game_goals():
    """Create a basic set of goals for completing Elden Ring"""
    
    goals = GoalSystem()
    
    # STARTING AREA: Chapel of Anticipation
    chapel_goal = GameGoal(
        name="Escape Chapel of Anticipation",
        description="""The Chapel of Anticipation is the starting tutorial/boss arena.
        
        Objective:
        - Find the boss arena in the Chapel
        - Enter the arena to spawn the Grafted Scion (tutorial boss)
        - Either defeat the boss OR die to it
        
        Exit conditions:
        - Beating the boss: Teleports you to Limgrave with items
        - Dying to the boss: You resurrect at the starting Site of Grace
        
        Both outcomes are acceptable for escaping this starting area.
        The key is to find the arena and trigger the boss fight.
        
        Strategy:
        - Explore the Chapel carefully
        - Look for a fog gate or arena entrance
        - When you find it, enter to spawn the Grafted Scion
        - Engage in combat (winning or losing both lead to leaving the area)""",
        priority=12,  # HIGHEST - must escape starting area first
        reward_value=1500,
        required_sub_goals=2
    )
    
    chapel_goal.add_sub_goal("Navigate Chapel of Anticipation", reward=500)
    chapel_goal.add_sub_goal("Find and enter the boss arena (spawn Grafted Scion)", reward=1000)
    
    goals.add_goal(chapel_goal)
    
    # PRIMARY OBJECTIVE 1: Defeat two Shardbearer bosses to access Leyndell
    main_goal = GameGoal(
        name="Defeat Two Shardbearers",
        description="""Defeat any 2 of the following 5 Shardbearer bosses to gain access to Leyndell:
        - Godrick the Grafted (requires defeating Margit, the Fell Omen first)
        - Rennala, Queen of the Full Moon (requires finding Academy Key and defeating Red Wolf of Radagon)
        - Starscourge Radahn (no prerequisites)
        - Rykard, Lord of Blasphemy (no prerequisites)
        - Mohg, Lord of Blood (requires completing White Mask Varré's questline)""",
        priority=10,
        reward_value=2000,
        required_sub_goals=2  # Only need 2 of 5
    )
    
    # Add sub-goals for each Shardbearer path
    # Godrick path
    main_goal.add_sub_goal("Defeat Margit, the Fell Omen", reward=300)
    main_goal.add_sub_goal("Defeat Godrick the Grafted", reward=700)
    
    # Rennala path
    main_goal.add_sub_goal("Find Academy Key", reward=200)
    main_goal.add_sub_goal("Defeat Red Wolf of Radagon", reward=300)
    main_goal.add_sub_goal("Defeat Rennala, Queen of the Full Moon", reward=700)
    
    # Radahn path (no prerequisites)
    main_goal.add_sub_goal("Defeat Starscourge Radahn", reward=800)
    
    # Rykard path (no prerequisites)
    main_goal.add_sub_goal("Defeat Rykard, Lord of Blasphemy", reward=700)
    
    # Mohg path
    main_goal.add_sub_goal("Complete White Mask Varré's Questline", reward=300)
    main_goal.add_sub_goal("Defeat Mohg, Lord of Blood", reward=700)
    
    goals.add_goal(main_goal)
    
    # PRIMARY OBJECTIVE 2: Access Leyndell, Royal Capital
    leyndell_goal = GameGoal(
        name="Access Leyndell, Royal Capital",
        description="""Access the Royal Capital by defeating the Draconic Tree Sentinel at the entrance or completing Fia's questline.
        Choose one of two paths:
        - Path A: Defeat Draconic Tree Sentinel (direct combat)
        - Path B: Complete Fia's Questline (longer questline)""",
        priority=9,
        reward_value=1500,
        required_sub_goals=1  # Only need 1 path completed
    )
    
    # Add sub-goals for accessing Leyndell
    leyndell_goal.add_sub_goal("Defeat Draconic Tree Sentinel", reward=1200)
    leyndell_goal.add_sub_goal("Complete Fia's Questline", reward=800)
    
    goals.add_goal(leyndell_goal)
    
    # PRIMARY OBJECTIVE 3: Progress through the Capital
    capital_goal = GameGoal(
        name="Progress Through the Capital",
        description="""Defeat the bosses guarding the capital and progress toward the Elden Ring.
        Required bosses:
        - Defeat Godfrey, First Elden Lord (Golden Shade form)
        - Defeat Morgott, the Omen King""",
        priority=9,
        reward_value=2500,
        required_sub_goals=2  # Both required
    )
    
    # Add sub-goals for capital progression
    capital_goal.add_sub_goal("Defeat Godfrey, First Elden Lord", reward=1000)
    capital_goal.add_sub_goal("Defeat Morgott, the Omen King", reward=1000)
    
    goals.add_goal(capital_goal)
    
    # PRIMARY OBJECTIVE 4: Reach the Mountaintops of the Giants
    mountaintops_goal = GameGoal(
        name="Reach the Mountaintops of the Giants",
        description="""Access the Mountaintops and defeat the Fire Giant.
        Required steps:
        - Obtain the Rold Medallion (can be found or acquired through quests)
        - Use the Grand Lift of Rold
        - Defeat the Fire Giant""",
        priority=8,
        reward_value=2000,
        required_sub_goals=3  # All required
    )
    
    # Add sub-goals for Mountaintops progression
    mountaintops_goal.add_sub_goal("Obtain the Rold Medallion", reward=500)
    mountaintops_goal.add_sub_goal("Activate Grand Lift of Rold", reward=500)
    mountaintops_goal.add_sub_goal("Defeat Fire Giant", reward=1000)
    
    goals.add_goal(mountaintops_goal)
    
    # PRIMARY OBJECTIVE 5: Go to Crumbling Farum Azula
    farum_azula_goal = GameGoal(
        name="Go to Crumbling Farum Azula",
        description="""Burn the Erdtree and enter Crumbling Farum Azula.
        Required steps:
        - Use the Forge of the Giants to burn the Erdtree
        - Defeat the Godskin Duo
        - Defeat Beast Clergyman / Maliketh, the Black Blade""",
        priority=8,
        reward_value=2500,
        required_sub_goals=3  # All required
    )
    
    # Add sub-goals for Farum Azula progression
    farum_azula_goal.add_sub_goal("Burn the Erdtree at Forge of the Giants", reward=500)
    farum_azula_goal.add_sub_goal("Defeat the Godskin Duo", reward=1000)
    farum_azula_goal.add_sub_goal("Defeat Maliketh, the Black Blade", reward=1000)
    
    goals.add_goal(farum_azula_goal)
    
    # PRIMARY OBJECTIVE 6: Return to Leyndell (Ashen Capital) and finish the game
    endgame_goal = GameGoal(
        name="Return to Leyndell (Ashen Capital) and Finish the Game",
        description="""Complete the final boss sequence in the Ashen Capital.
        Required bosses:
        - Defeat Sir Gideon Ofnir, the All-Knowing
        - Defeat Godfrey, First Elden Lord / Hoarah Loux, Warrior (2-phase fight)
        - Defeat Radagon of the Golden Order / Elden Beast (2-phase fight)""",
        priority=10,
        reward_value=5000,
        required_sub_goals=3  # All required
    )
    
    # Add sub-goals for final boss sequence
    endgame_goal.add_sub_goal("Defeat Sir Gideon Ofnir, the All-Knowing", reward=1500)
    endgame_goal.add_sub_goal("Defeat Godfrey / Hoarah Loux (2-phase fight)", reward=1750)
    endgame_goal.add_sub_goal("Defeat Radagon / Elden Beast (2-phase fight)", reward=1750)
    
    goals.add_goal(endgame_goal)
    
    # CRITICAL OBJECTIVE: Interact with Melina at Bonfires
    melina_goal = GameGoal(
        name="Interact with Melina at Bonfires",
        description="""Melina is a crucial NPC who appears as a dialogue option at bonfires/Sites of Grace.
        
        How she appears:
        - Rest at any bonfire/Site of Grace
        - Melina appears as a menu option to talk to
        - Select her dialogue option to advance the story
        
        What she gives:
        - TORRENT (the MOUNT) - Only available after talking to Melina
        - STORY PROGRESSION
        - LEVELING SERVICES
        
        CRITICAL: The mount CANNOT be summoned until you receive Torrent from Melina.
        DO NOT attempt to use the mount (Summon Mount action) before getting it from her.
        Attempting to summon the mount before getting it is a waste and penalized.
        
        Strategy for AI:
        - Rest at bonfires frequently (especially early game)
        - When Melina's dialogue option appears, select it
        - Always choose affirmative responses to her questions
        - After first meeting: You will receive Torrent (the mount item)
        - Only after receiving Torrent can you summon it
        
        The AI should prioritize talking to Melina whenever possible.
        She is essential for story progression and gaining access to the mount.""",
        priority=11,  # HIGHEST priority - more important than boss fights
        reward_value=3000,
        required_sub_goals=1
    )
    
    melina_goal.add_sub_goal("Talk to Melina at a bonfire menu", reward=1500)
    melina_goal.add_sub_goal("Answer Melina's questions affirmatively", reward=1500)
    
    goals.add_goal(melina_goal)
    
    # CRITICAL OBJECTIVE: Learn Fast Travel Mechanics
    fast_travel_goal = GameGoal(
        name="Master Fast Travel Mechanics",
        description="""Fast travel is ONLY useful when you have multiple bonfires unlocked.
        
        Requirements for fast travel:
        - MINIMUM 2 bonfires must be activated/rested at
        - Cannot fast travel if you only have 1 bonfire
        - Cannot fast travel while in combat
        - Cannot fast travel to a bonfire you haven't activated yet
        
        When to use fast travel:
        - When stuck with no way forward (use sparingly)
        - When trapped in a dead end like Roundtable Hold
        - To quickly travel between distant areas (only when necessary)
        
        How fast travel works:
        - Open the map with G key
        - Select any previously activated bonfire/Site of Grace
        - You will be transported there (outside combat only)
        
        IMPORTANT:
        - Don't open the map constantly - it's rarely needed
        - Focus on exploring naturally without fast travel
        - Only use fast travel when you MUST escape an area
        - Setting map markers is NOT useful - ignore that feature
        
        Early game strategy:
        - You will likely only have 1 bonfire at the start
        - Can't use fast travel yet - just explore normally
        - Get a second bonfire activated, then you can fast travel if needed""",
        priority=10,
        reward_value=2000,
        required_sub_goals=2
    )
    
    fast_travel_goal.add_sub_goal("Activate second bonfire for fast travel access", reward=500)
    fast_travel_goal.add_sub_goal("Use fast travel to escape a dead end", reward=1500)
    
    goals.add_goal(fast_travel_goal)
    
    # IMPORTANT OBJECTIVE: Understand Locked Doors and Barriers
    locked_doors_goal = GameGoal(
        name="Navigate Locked Doors and Barriers",
        description="""Some doors are locked or blocked and cannot be opened directly.
        
        Types of barriers:
        - Locked doors: Require a key item to unlock
        - Blocked passages: Require pulling a lever or switch
        - Sealed gates: May require defeating a nearby boss or NPC
        - Fog walls: Some doors have fog barriers that must be passed through
        
        How to handle locked doors:
        1. If a door won't open, it's likely locked or blocked
        2. Search the surrounding area for:
           - Keys lying on the ground
           - Levers to pull (interact with)
           - Switches to activate
           - NPCs that might have keys
        3. If nothing found nearby, explore other areas first
        4. Some doors only unlock after progressing the story or defeating a boss
        
        Strategy for AI:
        - Attempt to interact with doors (press E)
        - If blocked, explore alternative routes
        - Look for keys and levers in adjacent areas
        - Come back later after progressing further
        - Use fast travel to explore other areas when stuck
        
        Common locations with locked doors:
        - Dungeons: Often have multiple locked doors requiring progression
        - Catacombs: Boss doors are usually locked until you reach them
        - Story areas: Some progress gates require keys or boss defeats
        
        Priority: When unable to progress due to a locked door, 
        explore other areas rather than getting stuck.""",
        priority=7,
        reward_value=1000,
        required_sub_goals=2
    )
    
    locked_doors_goal.add_sub_goal("Find a key to unlock a door", reward=400)
    locked_doors_goal.add_sub_goal("Pull a lever to unblock a passage", reward=600)
    
    goals.add_goal(locked_doors_goal)
    
    # Side objectives
    goals.add_goal(GameGoal(
        name="Survive Combat",
        description="Stay alive in combat without taking excessive damage",
        priority=8,
        reward_value=500
    ))
    
    goals.add_goal(GameGoal(
        name="Learn Attack Patterns",
        description="Learn and dodge enemy attack patterns",
        priority=7,
        reward_value=300
    ))
    
    goals.add_goal(GameGoal(
        name="Manage Resources",
        description="Manage HP, Stamina, and Magic efficiently",
        priority=6,
        reward_value=200
    ))
    
    goals.add_goal(GameGoal(
        name="Explore and Navigate",
        description="Navigate through dungeons and reach new areas",
        priority=5,
        reward_value=100
    ))
    
    return goals


def create_custom_goals():
    """Interactively create custom goals"""
    
    goals = GoalSystem()
    
    print("\n" + "=" * 60)
    print("CUSTOM GOAL CREATION")
    print("=" * 60)
    print("Define what the AI needs to accomplish.\n")
    
    while True:
        name = input("Goal name (or 'done' to finish): ").strip()
        if name.lower() == 'done':
            break
        
        description = input("Description: ").strip()
        priority = int(input("Priority (1-10, higher = more important): ") or "5")
        reward = int(input("Reward points for completing: ") or "100")
        
        goal = GameGoal(name, description, priority, reward)
        goals.add_goal(goal)
        print(f"✓ Goal '{name}' added!\n")
    
    return goals


if __name__ == "__main__":
    print("Goal System Demo\n")
    
    # Create base game goals
    goals = create_base_game_goals()
    goals.list_goals()
    
    # Set main objective
    goals.set_main_objective("Defeat Godrick the Grafted")
    
    print(f"\nCurrent goal reward value: {goals.get_goal_reward()} points")
