import hashlib
import os
import time
import gymnasium as gym
import requests
from bs4 import BeautifulSoup

from . import constants
from .prompts import PromptTemplates
from .llm_client import LLMClient


def clean_str(p):
    return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


# --------------------------------------------------------------------------
# Wikipedia response cache -- off by default, enabled with WIKI_CACHE=1.
#
# The 3-arm isolation gate compares realized trajectories across arms that run
# minutes apart. Wikipedia is live: an edit between two arms would surface as a
# trajectory divergence that has nothing to do with speculation isolation. The
# cache pins the corpus so the only thing varying across arms is the thing
# under test. WIKI_CACHE_STATS lets the gate assert that every arm after the
# first is a pure cache hit -- a nonzero miss count means the comparison is
# confounded and the result must not be reported.
#
# Keyed by full search URL. Persisted under WIKI_CACHE_DIR so the gate is
# re-runnable across processes.
# --------------------------------------------------------------------------
WIKI_CACHE_ENABLED = os.environ.get("WIKI_CACHE", "0") == "1"
WIKI_CACHE_DIR = os.environ.get("WIKI_CACHE_DIR", "/tmp/specmem/wiki_cache")
WIKI_CACHE_STATS = {"hits": 0, "misses": 0}
_WIKI_CACHE_MEM = {}


def reset_wiki_cache_stats():
    WIKI_CACHE_STATS["hits"] = 0
    WIKI_CACHE_STATS["misses"] = 0


def _wiki_cache_path(url):
    return os.path.join(WIKI_CACHE_DIR, hashlib.sha256(url.encode()).hexdigest() + ".html")


def wiki_get(url, headers):
    """GET a Wikipedia URL, optionally through the pinned cache."""
    if not WIKI_CACHE_ENABLED:
        WIKI_CACHE_STATS["misses"] += 1
        return requests.get(url, headers=headers).text

    if url in _WIKI_CACHE_MEM:
        WIKI_CACHE_STATS["hits"] += 1
        return _WIKI_CACHE_MEM[url]

    path = _wiki_cache_path(url)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        _WIKI_CACHE_MEM[url] = text
        WIKI_CACHE_STATS["hits"] += 1
        return text

    WIKI_CACHE_STATS["misses"] += 1
    text = requests.get(url, headers=headers).text
    os.makedirs(WIKI_CACHE_DIR, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)
    _WIKI_CACHE_MEM[url] = text
    return text


class TextSpace(gym.spaces.Space):
    def contains(self, x) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        return isinstance(x, str)


class WikiEnv(gym.Env):

    def __init__(self, guess_model_name):
        super().__init__()
        self.page = None
        self.obs = None
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = None
        self.steps = 0
        self.answer = None
        self.observation_space = self.action_space = TextSpace()
        self.search_time = 0
        self.num_searches = 0
        self.sim_obs = None
        self.guess_model_name = guess_model_name
        self.guess_llm = LLMClient(
            model_name=guess_model_name,
            temperature=constants.guess_temperature,
            max_tokens=constants.max_guess_output_tokens,
            top_p=constants.guess_top_p,
            base_url=constants.spec_base_url,
        )

    def _get_obs(self):
        return self.obs

    def _get_info(self):
        return {"steps": self.steps, "answer": self.answer}

    def reset(self, seed=None, return_info=False, options=None):
        self.obs = ("Interact with Wikipedia using search[], lookup[], and "
                    "finish[].\n")
        self.page = None
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = None
        self.steps = 0
        self.answer = None
        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def construct_lookup_list(self, keyword):
        if self.page is None:
            return []
        paragraphs = self.page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        sentences = []
        for p in paragraphs:
            sentences += p.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        parts = [p for p in sentences if keyword.lower() in p.lower()]
        return parts

    @staticmethod
    def get_page_obs(page):
        paragraphs = page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        sentences = []
        for p in paragraphs:
            sentences += p.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        return ' '.join(sentences[:5])

    def guess_step(self, entity, simulate=False, max_retries=3):
        prompt_wrap = PromptTemplates.GUESS_STEP_PROMPT.format(entity)
        llm_response = None
        for _ in range(max_retries):
            llm_response = self.guess_llm.call(prompt_wrap)
            if llm_response:
                break
        if not llm_response:
            llm_response = ""
        if simulate:
            self.sim_obs = self.get_page_obs(llm_response)
        self.page = llm_response
        self.obs = self.get_page_obs(self.page)

    def search_step(self, entity):
        entity_ = entity.replace(" ", "+")
        search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
        old_time = time.time()
        requests_headers = {"User-Agent": "React/1.0"}
        response_text = wiki_get(search_url, requests_headers)
        self.search_time += time.time() - old_time
        self.num_searches += 1
        soup = BeautifulSoup(response_text, features="html.parser")
        result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
        if result_divs:
            self.result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
            self.obs = f"Could not find {entity}. Similar: {self.result_titles[:5]}."
        else:
            page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
            if any("may refer to:" in p for p in page):
                self.search_step("[" + entity + "]")
            else:
                self.page = ""
                for p in page:
                    if len(p.split(" ")) > 2:
                        self.page += clean_str(p)
                        if not p.endswith("\n"):
                            self.page += "\n"
                self.obs = self.get_page_obs(self.page)
                self.lookup_keyword = self.lookup_list = self.lookup_cnt = None

    def step(self, action, step_type="wiki"):
        reward = 0
        done = False
        action = action.strip()
        if self.answer is not None:
            done = True
            return self.obs, reward, done, self._get_info()

        if action.startswith("search[") and action.endswith("]"):
            entity = action[len("search["):-1]
            if step_type == "wiki":
                self.search_step(entity)
            elif step_type == "guess":
                self.guess_step(entity)
            elif step_type == "simulate":
                self.guess_step(entity, simulate=True)
            else:
                raise ValueError("Run is not valid. Step type needs to be wiki or guess")
        elif action.startswith("lookup[") and action.endswith("]"):
            keyword = action[len("lookup["):-1]
            if self.lookup_keyword != keyword:
                self.lookup_keyword = keyword
                self.lookup_list = self.construct_lookup_list(keyword)
                self.lookup_cnt = 0
            if self.lookup_cnt >= len(self.lookup_list):
                self.obs = "No more results.\n"
            else:
                self.obs = f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) " + self.lookup_list[self.lookup_cnt]
                self.lookup_cnt += 1
        elif action.startswith("finish[") and action.endswith("]"):
            answer = action[len("finish["):-1]
            self.answer = answer
            done = True
            self.obs = f"Episode finished, reward = {reward}\n"
        elif action.startswith("think[") and action.endswith("]"):
            self.obs = "Nice thought."
        else:
            self.obs = "Invalid action: {}".format(action)

        self.steps += 1
        return self.obs, reward, done, self._get_info()

    def get_time_info(self):
        speed = self.search_time / self.num_searches if self.num_searches else 0
        return {
            "call_speed": speed,
            "call_time": self.search_time,
            "num_calls": self.num_searches,
        }
