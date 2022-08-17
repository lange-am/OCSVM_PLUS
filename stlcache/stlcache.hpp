//
// Copyright (C) 2011-2018 Denis V Chapligin, Martin Hrabovsky, Vojtech Ondruj
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
// Changed by Andrey M. Lange
//


#ifndef STLCACHE_STLCACHE_HPP_INCLUDED
#define STLCACHE_STLCACHE_HPP_INCLUDED

#ifdef _MSC_VER
#pragma warning( disable : 4290 )
#endif /* _MSC_VER */

#include <stdexcept>
#include <typeinfo>
#include <map>

//#include <iostream>

using namespace std;

#include <stlcache/exceptions.hpp>
#include <stlcache/policy.hpp>
#include <stlcache/policy_lru.hpp>
#include <stlcache/policy_lfu.hpp>

namespace stlcache {

    template<
        class Key,
        class Data,
        class Policy,
        class Compare = less<Key>,
        template <typename T> class Allocator = allocator>
    class cache {
        typedef map<Key, Data, Compare, Allocator<pair<const Key, Data> > > storageType;
         storageType _storage;
         std::size_t _maxEntries;
         std::size_t _currEntries;
         typedef typename Policy::template bind<Key, Allocator> policy_type;
         policy_type* _policy;
         Allocator<policy_type> policyAlloc;

    public:
        /*! \brief The Key type
         */
        typedef Key                                                                 key_type;
        /*!  \brief The Data type
         */
        typedef Data                                                                mapped_type;
        /*! \brief Combined is_found, value type
          */
        typedef pair<bool, Data>                                                    find_pair_type; // added by A. Lange
        /*! \brief Combined key, value type
          */
        typedef pair<const Key, Data>                                               value_type;
        /*! \brief Compare type used by this instance
          */
        typedef Compare                                                             key_compare;
        /*! \brief Allocator type used by this instance
          */
        typedef Allocator<pair<const Key, Data> >                                   allocator_type;
        /*! \brief Nested class to compare elements (see member function value_comp)
          */
        typedef typename storageType::value_compare                                 value_compare;
        /*! \brief Allocator::iterator
          */
        typedef typename storageType::iterator                                      iterator;
        /*! \brief Allocator::const_iterator
          */
        typedef typename storageType::const_iterator                                const_iterator;
        /*! \brief Allocator::reference
          */
        typedef typename storageType::reference                                     reference;
        /*! \brief Allocator::const_reference
          */
        typedef typename storageType::const_reference                               const_reference;
        /*! \brief Type used for storing object sizes, specific to a current platform (usually a size_t)
          */
        typedef typename storageType::size_type                                     size_type;
        /*! \brief Type used for address calculations, specific to a current platform (usually a ptrdiff_t)
          */
        typedef typename storageType::difference_type                               difference_type;
        /*! \brief Allocator::pointer
          */
        typedef typename storageType::pointer                                       pointer;
        /*! \brief Allocator::const_pointer
          */
        typedef typename storageType::const_pointer                                 const_pointer;

        /*! \name std::map interface wrappers
         *  Simple wrappers for std::map calls, that we are using only for mimicking the map interface
         */
        //@{
        /*! \brief Allocator object accessor
          * 
          *  Provides access to the allocator object, used to constuct the container.
          *
          *  \return The allocator object of type Allocator<pair<const Key, Data> >.
          */
        allocator_type get_allocator() const {
            return _storage.get_allocator();
        }

        /*! \brief Value comparision object accessor
         *
         *   Provides access to a comparison object that can be used to compare two element values (pairs) to get whether the key of the first goes before the second. 
         *   The mapped value, although part of the pair, is not taken into consideration in this comparison - only the key value.
         *
         */
        value_compare value_comp() const {
            return _storage.value_comp();
        }

        /*! \brief Key comparision object accessor
          *
          *  Provides access to the  comparison object associated with the container, which can be used to compare the keys of two elements in the container.
          *  This comparison object is set on object construction, and may either be a pointer to a function or an object of a class with a function call operator.
          *  In both cases it takes two arguments of the same type as the element keys,
          *  and returns true if the first argument is considered to go before the second in the strict weak ordering for the map object,
          *  and false otherwise.
          *  Notice that the returned comparison object takes as parameters the element keys themselves, not entire element values (pairs).
          *
          *  \return The key comparison object of type key_compare, defined to Compare, which is the fourth template parameter in the map class template.
          */
        key_compare key_comp() const {
            return _storage.key_comp();
        }

        /*! \brief Test whether cache is empty
          *
          *  Tells whether the map container is empty, i.e. whether its size is 0. This function does not modify the content of the container in any way. 
          *  In terms of performance it is not equal to call
          *  \code
          *      cache.size()==0
          *  \endcode
          *
          *  \return true if the container size is 0, false otherwise.
          *
          *  \see size
          */
        bool empty() const {
            return _storage.empty();
        }
        //@}

        /*! \name cache api
         *  members that are specific to cache or implemented with some cache specific things
         */
        //@{
        /*!
         * \brief Clear the cache
         *
         * Removes all cache entries, drops all usage count data and so on.
         *
         * \see size
         * \see empty
         *
         */
        void clear() {
            _storage.clear();
            _policy->clear();
            this->_currEntries=0;
        }

        /*!
         * \brief Swaps contents of two caches
         *
         * Exchanges the content of the cache with the content of mp, which is another cache object containing elements of the same type and using the same expiration policy.
         * Sizes may differ. Maximum number of entries may differ too.
         *
         * \param <mp> Another cache of the same type as this whose cache is swapped with that of this cache.
         *
         * \throw <exception_invalid_policy> Thrown when the policies of the caches to swap are incompatible.
         *
         * \see cache::operator=
         */
        void swap ( cache<Key,Data,Policy,Compare,Allocator>& mp ) {
            _storage.swap(mp._storage);
            _policy->swap(*mp._policy);

            std::size_t m=this->_maxEntries;
            this->_maxEntries=mp._maxEntries;
            mp._maxEntries=m;

            this->_currEntries=this->_size();
            mp._currEntries=mp.size();
        }

        /*!
         * \brief Removes a entry from cache
         *
         * The entry with the specified key value will be removed from cache and it's usage count information will be erased. Size of the cache will be decreased.
         * For non-existing entries nothing will be done.
         *
         * \param <_k> Key to remove.
         *
         * \return 1 when entry is removed (ie number of removed emtries, which is always 1, as keys are unique) or zero when nothing was done.
         */
        size_type erase(const Key& _k) {
            return this->_erase(_k);
        }

        /*!
         * \brief Insert element to the cache
         *
         * The cache is extended by inserting a single new element. This effectively increases the cache size. 
         * Because cache do not allow for duplicate key values, the insertion operation checks for each element inserted whether another element exists already in the container with the same key value.
         * If so, the element is not inserted and its mapped value is not changed in any way.
         * Extension of cache could result in removal of some elements, depending of the cache fullness and used policy. 
         * It is also possible, that removal of excessive entries will fail, therefore insert operation will fail too.
         *
         * \throw <exception_cache_full>  Thrown when there are no available space in the cache and policy doesn't allows removal of elements.
         * \throw <exception_invalid_key> Thrown when the policy doesn't accepts the key
         *
         * \return true if the new elemented was inserted or false if an element with the same key existed.
         */
        bool insert(Key _k, Data _d) {
            while (this->_currEntries >= this->_maxEntries) {
                _victim<Key> victim=_policy->victim();
                if (!victim) {
                    throw exception_cache_full("The cache is full and no element can be expired at the moment. Remove some elements manually");
                }
                this->_erase(*victim);
            }

            _policy->insert(_k);

            bool result=_storage.insert(value_type(_k,_d)).second;
            if (result) {
                _currEntries++;
            }

            return result;
        }

        /*
         * \bried Merge two caches
         *
         * Inserts items, missing in *this, but existing in the second to *this.
         * For the existing items, reference count will be increased.
         */
        void merge(const cache<Key, Data, Policy, Compare, Allocator>& second){
            for (auto it = second._storage.begin(); it != second._storage.end(); it++) {
                if (!this->check(it->first)) {
                    this->insert(it->first, it->second);
                } else {
                    this->touch(it->first);
                }
            }
        }

        /*!
         * \brief Maximum cache size accessor
         *
         * Returns the maximum number of elements that the cache object can hold. This is the maximum potential size the cache can reach. 
         * It is specified at construction time and cannot be modified (with excaption of \link cache::swap swap operation \endlink)
         *
         * \return The maximum number of elements a cache can have as its content.
         *
         * \see size
         */
        size_type max_size() const {
            return this->_maxEntries;
        }

        /*! \brief Counts entries in the cache.
          *
          *  Provides information on number of cache entries. For checking, whether the cache empty or not, please use the \link cache::empty empty function \endlink
          *
          *  \return Number of object in the cache (size of cache)
          *
          *  \see empty
          *  \see clear
          */
        size_type size() const {
            return this->_size();
        }

        /*! \brief Checks whether the elements is in the cache without touching the entry usage count.
         * 
         *   Policy does not take into account that this was the element of interest.
         * 
         *   \return A pair <(bool) does element present, data value>.
         * 
         *   \author A.Lange (15 June 2021)
         */
        find_pair_type find(const Key& _k) const {
            const_iterator it = _storage.find(_k);
            return find_pair_type(it!=_storage.end(), it->second);
        }

        /*! \brief Tries to find the key in cache and returns a pair <element presence, data value>.
         * 
         *  Also touches entry's usage count, i.e. records the need for the element in the policy.
         *
         *  Possible usage:
         *  \code 
         *      cache::find_pair_type p = cache.recorded_find(key)
         *      if !p.first {
         *          calculated_value = ...  // need to spend resources for new calculation
         *          cache.insert(key, calculated_value)
         *      }
         *  \endcode
         * 
         *  \author A.Lange (15 June 2021)
         */
        find_pair_type recorded_find(const Key& _k) {
            return this->_recorded_find(_k);
        }
        //@}

        //@{
        /*!
         * \brief Copy cache content
         *
         * Assigns a copy of the elements in other as the new content for the cache. Usage counts for entries are copied too.
         * The elements contained in the object before the call are dropped, and replaced by copies of those in cache x, if any.
         * After a call to this member function, both the map object and x will have the same size and compare equal to each other.
         *
         */
        cache<Key, Data, Policy, Compare, Allocator>& operator= (const cache<Key, Data, Policy, Compare, Allocator>& other) {
            this->_storage=other._storage;
            this->_maxEntries=other._maxEntries;
            this->_currEntries=this->_storage.size();

            policy_type localPolicy(*other._policy);
            this->_policy = policyAlloc.allocate(1);
            policyAlloc.construct(this->_policy, localPolicy);
            return *this;
        }

        /*!
         * \brief A copy constructor
         *
         *  The object is initialized to have the same contents and properties as the other cache object.
         *
         *  \param <other> a cache object with the same template parameters
         */
        cache(const cache<Key, Data, Policy, Compare, Allocator>& other) {
            *this=other;
        }
        /*!
         * \brief Primary constructor.
         *
         * Constructs an empty cache object and sets a maximum size for it. It is the only way to set size for a cache and it can't be changed later.
         * You could also  pass optional comparator object, compatible with Compare.
         *
         * \param <size> Maximum number of entries, allowed in the cache.
         * \param <comp> Comparator object, compatible with Compare type. Defaults to Compare()
         *
         */
        explicit cache(const size_type size, const Compare& comp = Compare()) {
            this->_storage=storageType(comp, Allocator<pair<const Key, Data> >());
            this->_maxEntries=size;
            this->_currEntries=0;

            policy_type localPolicy(size);
            this->_policy = policyAlloc.allocate(1);
            policyAlloc.construct(this->_policy,localPolicy);
        }

        /*!
         * \brief destructor
         *
         * Destructs the cache object. This calls each of the cache element's destructors, and deallocates all the storage capacity allocated by the cache.
         *
         */
        ~cache() {
            policyAlloc.destroy(this->_policy);
            policyAlloc.deallocate(this->_policy,1);
        }
        //@}
    protected:
        // added by A.Lange
        find_pair_type _recorded_find(const Key& _k) throw() {
            _policy->touch(_k);
            const_iterator it = _storage.find(_k);
            return find_pair_type (it!=_storage.end(), it->second);
        }
        size_type _erase(const Key& _k) throw() {
            size_type ret=_storage.erase(_k);
            _policy->remove(_k); // removing entry from policy storage. Policy also forgets the entry!

            _currEntries-=ret;

            return ret;
        }
        size_type _size() const throw() {
            assert(this->_currEntries==_storage.size());
            return this->_currEntries;
        }
    };
}

#endif /* STLCACHE_STLCACHE_HPP_INCLUDED */
